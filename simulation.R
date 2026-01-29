# ============================================================
# Li et al. (Biometrics) simulation reproduction (NO packages)
# - Methods: Mix-L, Mix-AL, Mix-HP-L, Mix-HP-AL
# - Selection: BIC over m in {2,3,4} and lambda grid (m-specific)
# - EM + coordinate descent (weighted lasso)
# - HP uses equality constraint sum_j B[k,j]=0 enforced by eliminating B_m
#   BUT penalty still includes |B_mk| explicitly (two-abs 1D update)
#
# IMPORTANT FIXES / IMPROVEMENTS:
# 1) EM ll_old update bug fixed
# 2) Mix-HP partial_m sign bug fixed
# 3) HP df in BIC uses Bfree (independent params) not Bfull
# 4) FPR/TPR selection uses PRE-matching B_hat (recommended)
# 5) simulate_dataset mu vectorized
#
# NEW (stability-critical):
# 6) rho update uses CLOSED-FORM M-step under rescaled loglik (paper model (2))
#    => prevents sigma2 blow-up when component weight small
# 7) strict component death handling + rho/sigma2 clamps
# 8) HP M-step rewritten to manage eta (=X %*% beta) with periodic full recompute
# 9) Adaptive weights built from ridge-WLS initializer (avoid "0 -> infinite weight")
# 10) lambda grid is m-specific: lam_max / m (mixture responsibilities scale)
# ============================================================


# -------------------------
# Basic helpers
# -------------------------
soft_thresh <- function(z, g) {
  if (z >  g) return(z - g)
  if (z < -g) return(z + g)
  0
}

logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

count_nz <- function(v, tol = 1e-8) sum(abs(v) > tol)


# -------------------------
# Covariance matrices
# -------------------------
cov_mat <- function(p, correlated = FALSE) {
  if (!correlated) return(diag(p))
  toeplitz(0.5^(0:(p - 1)))
}


# -------------------------
# SNR <-> delta mapping (paper)
# SNR = 25 / delta  => delta = 25 / SNR
# -------------------------
delta_from_snr <- function(snr) 25 / snr


# -------------------------
# True scaled betas (paper setting)
# p0 = 10 relevant; among them p00 = 3 hetero (positions 8,9,10)
# beta0 common: (1..1,0,0,0)/sqrt(delta)
# hetero patterns across components:
# -------------------------
true_scaled_betas <- function(p, delta) {
  stopifnot(p >= 10)
  beta0  <- rep(0, p)
  beta1  <- rep(0, p)
  beta2  <- rep(0, p)
  beta3  <- rep(0, p)
  
  beta0[1:10] <- c(rep(1, 7), rep(0, 3)) / sqrt(delta)
  
  beta1[1:10] <- c(rep(0, 7),  0, -3,  3) / sqrt(delta)
  beta2[1:10] <- c(rep(0, 7), -3,  3,  0) / sqrt(delta)
  beta3[1:10] <- c(rep(0, 7),  3,  0, -3) / sqrt(delta)
  
  list(beta0 = beta0,
       beta_het = cbind(beta1, beta2, beta3)) # p x 3
}


# -------------------------
# Data generator (paper model (3) -> rescaled model (2))
# Model: f(y|x)= sum_j pi_j * rho_j/sqrt(2pi) * exp(-1/2 (rho_j y - x^T beta_tilde_j)^2)
# beta_tilde_j = beta0 + beta_j (scaled)
# unscaled b_j = beta_tilde_j / rho_j
# sigma^2 = 1/rho^2, paper: sigma^2 = delta*(0.1,0.1,0.4)
# -------------------------
simulate_dataset <- function(n = 200, p = 60, snr = 50, correlated = FALSE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  delta <- delta_from_snr(snr)
  Sigma <- cov_mat(p, correlated)
  
  # X ~ N(0, Sigma)
  X <- matrix(rnorm(n * p), n, p)
  L <- chol(Sigma)
  X <- X %*% L
  
  m_true <- 3
  pi_true <- rep(1 / m_true, m_true)
  
  sigma2_true <- delta * c(0.1, 0.1, 0.4)
  sigma_true  <- sqrt(sigma2_true)
  rho_true    <- 1 / sigma_true
  
  bet <- true_scaled_betas(p, delta)
  beta0 <- bet$beta0
  beta_het <- bet$beta_het
  
  beta_tilde_true <- sapply(1:m_true, function(j) beta0 + beta_het[, j])
  beta_tilde_true <- matrix(beta_tilde_true, nrow = p, ncol = m_true)
  
  b_true <- sweep(beta_tilde_true, 2, rho_true, "/")
  
  z <- sample.int(m_true, size = n, replace = TRUE, prob = pi_true)
  
  # vectorized mu
  mu <- rowSums(X * t(b_true[, z, drop = FALSE]))
  y <- rnorm(n, mean = mu, sd = sigma_true[z])
  
  list(
    y = y, X = X, z = z,
    delta = delta, snr = snr, correlated = correlated,
    b_true = b_true,
    beta_tilde_true = beta_tilde_true,
    pi_true = pi_true,
    sigma2_true = sigma2_true,
    rho_true = rho_true,
    S_relevant = 1:10,
    S_common = 1:7,
    S_hetero = 8:10
  )
}


# -------------------------
# Mixture log-likelihood (scaled parameterization)
# beta_tilde_hat: p x m, rho_hat: m, pi_hat: m
# -------------------------
loglik_mix_scaled <- function(y, X, beta_tilde_hat, rho_hat, pi_hat) {
  n <- length(y)
  m <- length(pi_hat)
  ll <- 0
  for (i in 1:n) {
    lvec <- rep(NA_real_, m)
    for (j in 1:m) {
      mu <- sum(X[i, ] * beta_tilde_hat[, j])
      r  <- rho_hat[j] * y[i] - mu
      lvec[j] <- log(pi_hat[j] + 1e-300) + log(rho_hat[j] + 1e-300) - 0.5 * r * r - 0.5 * log(2*pi)
    }
    ll <- ll + logsumexp(lvec)
  }
  ll
}


# -------------------------
# E-step responsibilities (scaled)
# tau_ij proportional to pi_j * rho_j * exp(-1/2 (rho_j y - x beta)^2)
# -------------------------
estep_tau <- function(y, X, beta_tilde, rho, pi) {
  n <- length(y); m <- length(pi)
  logw <- matrix(0, n, m)
  for (j in 1:m) {
    mu <- as.vector(X %*% beta_tilde[, j])
    r  <- rho[j] * y - mu
    logw[, j] <- log(pi[j] + 1e-300) + log(rho[j] + 1e-300) - 0.5 * r * r
  }
  tau <- matrix(0, n, m)
  for (i in 1:n) {
    den <- logsumexp(logw[i, ])
    tau[i, ] <- exp(logw[i, ] - den)
  }
  tau
}


# -------------------------
# Weighted lasso by coordinate descent
# minimize 1/2 sum_i w_i (y_i - x_i^T b)^2 + lam * sum_k pen_w[k] * |b_k|
# -------------------------
lasso_cd_weighted <- function(X, y, w, lam, pen_w = NULL,
                              b_init = NULL,
                              maxit = 200, tol = 1e-6) {
  n <- nrow(X); p <- ncol(X)
  if (is.null(pen_w)) pen_w <- rep(1, p)
  if (is.null(b_init)) b <- rep(0, p) else b <- b_init
  
  r <- y - as.vector(X %*% b)
  a <- colSums((X^2) * w)
  
  for (it in 1:maxit) {
    b_old <- b
    for (k in 1:p) {
      if (a[k] < 1e-12) next
      xk <- X[, k]
      ck <- sum(w * xk * (r + xk * b[k]))
      bk_new <- soft_thresh(ck, lam * pen_w[k]) / a[k]
      d <- bk_new - b[k]
      if (abs(d) > 0) {
        b[k] <- bk_new
        r <- r - xk * d
      }
    }
    if (max(abs(b - b_old)) < tol) break
  }
  b
}


# -------------------------
# 1D update with TWO absolute terms (HP coordinate)
# minimize 0.5*a*b^2 - c*b + lam1*|b| + lam2*|b+s|
# -------------------------
update_two_abs_1d <- function(a, c, s, lam1, lam2) {
  if (a < 1e-12) return(0)
  
  cand <- c(0, -s)
  
  for (sgn1 in c(-1, 1)) {
    for (sgn2 in c(-1, 1)) {
      b <- (c - (lam1 * sgn1 + lam2 * sgn2)) / a
      ok1 <- (b > 0 && sgn1 == 1) || (b < 0 && sgn1 == -1) || abs(b) < 1e-12
      ok2 <- ((b + s) > 0 && sgn2 == 1) || ((b + s) < 0 && sgn2 == -1) || abs(b + s) < 1e-12
      if (ok1 && ok2) cand <- c(cand, b)
    }
  }
  
  f <- function(b) 0.5 * a * b * b - c * b + lam1 * abs(b) + lam2 * abs(b + s)
  vals <- sapply(cand, f)
  cand[which.min(vals)]
}


# -------------------------
# CLOSED-FORM rho update under rescaled Q:
# Q_j(rho) = A log rho - 0.5 * sum_i w_i (rho y_i - mu_i)^2
# derivative => B rho^2 - C rho - A = 0
#   A = sum w
#   B = sum w y^2
#   C = sum w y mu
# rho = (C + sqrt(C^2 + 4AB)) / (2B)
# -------------------------
update_rho_closed <- function(y, mu, w, rho_old,
                              w_min = 5,
                              rho_min = 1e-3, rho_max = 1e3) {
  A <- sum(w)
  if (!is.finite(A) || A < w_min) return(rho_old)
  
  B <- sum(w * (y^2))
  if (!is.finite(B) || B < 1e-12) return(rho_old)
  
  C <- sum(w * (y * mu))
  disc <- C^2 + 4 * A * B
  if (!is.finite(disc) || disc < 0) return(rho_old)
  
  rho <- (C + sqrt(disc)) / (2 * B)
  if (!is.finite(rho) || rho <= 0) rho <- rho_old
  
  rho <- max(min(rho, rho_max), rho_min)
  rho
}


# -------------------------
# Ridge-WLS initializer (for adaptive weights)
# b = argmin 0.5 sum w (y - Xb)^2 + 0.5 ridge ||b||^2
# -------------------------
ridge_wls <- function(X, y, w, ridge = 1e-4) {
  XtW <- t(X) * w
  A <- XtW %*% X
  diag(A) <- diag(A) + ridge
  b <- solve(A, XtW %*% y)
  as.vector(b)
}


# -------------------------
# M-step for Mix-L / Mix-AL
# - update pi (floored)
# - update beta_tilde_j via weighted lasso on (rho_j*y) ~ X
# - update rho via CLOSED-FORM (rescaled)
# -------------------------
mstep_mix_lasso <- function(y, X, tau, beta_tilde, rho, pi,
                            lam, pen_w_list,
                            max_cd = 200, tol_cd = 1e-6,
                            min_pi = 1e-6,
                            w_min = 5,
                            rho_min = 1e-3, rho_max = 1e3) {
  n <- length(y); m <- ncol(tau); p <- ncol(X)
  
  # pi update
  pi_new <- colMeans(tau)
  pi_new <- pmax(pi_new, min_pi)
  pi_new <- pi_new / sum(pi_new)
  
  beta_new <- beta_tilde
  rho_new <- rho
  
  # beta update (given rho)
  for (j in 1:m) {
    w <- tau[, j]
    yj <- rho_new[j] * y
    pen_w <- pen_w_list[[j]]
    beta_new[, j] <- lasso_cd_weighted(
      X, yj, w,
      lam = n * lam,
      pen_w = pen_w,
      b_init = beta_new[, j],
      maxit = max_cd, tol = tol_cd
    )
  }
  
  # rho update (closed form)
  for (j in 1:m) {
    w <- tau[, j]
    mu <- as.vector(X %*% beta_new[, j])
    rho_new[j] <- update_rho_closed(
      y = y, mu = mu, w = w, rho_old = rho_new[j],
      w_min = w_min, rho_min = rho_min, rho_max = rho_max
    )
  }
  
  sigma2_new <- (1 / (rho_new + 1e-12))^2
  list(beta_tilde = beta_new, rho = rho_new, pi = pi_new, sigma2 = sigma2_new)
}


# -------------------------
# M-step for Mix-HP-L / Mix-HP-AL
# parameters:
# - beta0 (p): common (scaled)
# - Bfree (p x (m-1)): free cluster-specific, Bm derived
# penalty includes |beta0k| + sum_j |B[k,j]| including derived m-th
#
# NUMERIC STABILITY:
# - manage eta_mat (= X %*% beta_tilde_j) and update it
# - periodically recompute eta_mat and rmat from scratch to avoid drift
# -------------------------
mstep_mix_hp <- function(y, X, tau, beta0, Bfree, rho, pi,
                         lam, pen_w0, pen_wB, # pen_wB: p x m weights on Bfull
                         max_cd = 200, tol_cd = 1e-6,
                         min_pi = 1e-6,
                         w_min = 5,
                         rho_min = 1e-3, rho_max = 1e3,
                         rebuild_every = 5) {
  n <- length(y); m <- ncol(tau); p <- ncol(X)
  stopifnot(ncol(Bfree) == m - 1)
  
  # pi update
  pi_new <- colMeans(tau)
  pi_new <- pmax(pi_new, min_pi)
  pi_new <- pi_new / sum(pi_new)
  
  rho_new <- rho
  beta0_new <- beta0
  Bfree_new <- Bfree
  
  make_Bfull <- function(Bf) {
    Bm <- -rowSums(Bf)
    cbind(Bf, Bm)
  }
  
  # Precompute columns for speed
  Xcols <- lapply(1:p, function(k) X[, k])
  
  # Build eta_mat and rmat from scratch
  rebuild_eta_r <- function(beta0v, Bf) {
    Bfull <- make_Bfull(Bf)
    eta <- matrix(0, n, m)
    for (j in 1:m) {
      bt <- beta0v + Bfull[, j]
      eta[, j] <- as.vector(X %*% bt)
    }
    r <- matrix(0, n, m)
    for (j in 1:m) r[, j] <- rho_new[j] * y - eta[, j]
    list(Bfull = Bfull, eta = eta, r = r)
  }
  
  tmp <- rebuild_eta_r(beta0_new, Bfree_new)
  Bfull <- tmp$Bfull
  eta_mat <- tmp$eta
  rmat <- tmp$r
  
  for (it in 1:max_cd) {
    maxchg <- 0
    
    # ---- update beta0 ----
    for (k in 1:p) {
      xk <- Xcols[[k]]
      a <- sum((xk^2) * rowSums(tau))
      if (a < 1e-12) next
      
      # cval = sum_j sum_i tau_ij * xk * (r_ij + xk * beta0_k)
      cval <- 0
      for (j in 1:m) {
        cval <- cval + sum(tau[, j] * xk * (rmat[, j] + xk * beta0_new[k]))
      }
      
      bnew <- soft_thresh(cval, n * lam * pen_w0[k]) / a
      d <- bnew - beta0_new[k]
      if (abs(d) > 0) {
        beta0_new[k] <- bnew
        maxchg <- max(maxchg, abs(d))
        # eta_j += xk*d  => r_j -= xk*d
        for (j in 1:m) {
          eta_mat[, j] <- eta_mat[, j] + xk * d
          rmat[, j] <- rmat[, j] - xk * d
        }
      }
    }
    
    # ---- update Bfree ----
    for (l in 1:(m - 1)) {
      for (k in 1:p) {
        xk <- Xcols[[k]]
        
        s_minus <- if ((m - 1) == 1) 0 else sum(Bfree_new[k, -l])
        bcur <- Bfree_new[k, l]
        
        a <- sum((tau[, l] + tau[, m]) * (xk^2))
        if (a < 1e-12) next
        
        # partial residuals excluding current coordinate:
        # comp l design = +xk => partial_l = r_l + xk*bcur
        partial_l <- rmat[, l] + xk * bcur
        
        # FIX (major): comp m design = (-xk) => partial_m = r_m + (-xk)*bcur = r_m - xk*bcur
        partial_m <- rmat[, m] - xk * bcur
        
        # cval = sum tau_l * xk * partial_l + tau_m * (-xk) * partial_m
        cval <- sum(tau[, l] * xk * partial_l + tau[, m] * (-xk) * partial_m)
        
        lam1 <- n * lam * pen_wB[k, l]
        lam2 <- n * lam * pen_wB[k, m]
        
        bnew <- update_two_abs_1d(a = a, c = cval, s = s_minus, lam1 = lam1, lam2 = lam2)
        d <- bnew - bcur
        if (abs(d) > 0) {
          Bfree_new[k, l] <- bnew
          maxchg <- max(maxchg, abs(d))
          
          # Effect on Bfull:
          # B_l increases by d => eta_l += xk*d => r_l -= xk*d
          eta_mat[, l] <- eta_mat[, l] + xk * d
          rmat[, l] <- rmat[, l] - xk * d
          
          # B_m = -sum(Bfree) decreases by d => eta_m -= xk*d => r_m += xk*d
          eta_mat[, m] <- eta_mat[, m] - xk * d
          rmat[, m] <- rmat[, m] + xk * d
        }
      }
    }
    
    # periodic rebuild to remove numeric drift
    if (it %% rebuild_every == 0) {
      tmp <- rebuild_eta_r(beta0_new, Bfree_new)
      Bfull <- tmp$Bfull
      eta_mat <- tmp$eta
      rmat <- tmp$r
    } else {
      Bfull <- make_Bfull(Bfree_new)
    }
    
    if (maxchg < tol_cd) break
  }
  
  # final Bfull
  Bfull <- make_Bfull(Bfree_new)
  
  # rho update (closed form)
  for (j in 1:m) {
    w <- tau[, j]
    mu <- as.vector(X %*% (beta0_new + Bfull[, j]))
    rho_new[j] <- update_rho_closed(
      y = y, mu = mu, w = w, rho_old = rho_new[j],
      w_min = w_min, rho_min = rho_min, rho_max = rho_max
    )
  }
  
  sigma2_new <- (1 / (rho_new + 1e-12))^2
  
  list(beta0 = beta0_new, Bfull = Bfull, Bfree = Bfree_new,
       rho = rho_new, pi = pi_new, sigma2 = sigma2_new)
}


# -------------------------
# Fit mixture model for fixed (m, lam) by EM (FIXED ll_old)
# Mix-L / Mix-AL
# -------------------------
fit_mix_L_or_AL <- function(y, X, m, lam,
                            adaptive = FALSE, gamma = 1,
                            max_em = 200, tol_em = 1e-5,
                            max_cd = 200, tol_cd = 1e-6,
                            n_start = 5, seed_base = 1,
                            min_pi = 1e-6,
                            w_min = 5,
                            rho_min = 1e-3, rho_max = 1e3,
                            w_cap = 1e4,
                            ridge_init = 1e-4) {
  n <- length(y); p <- ncol(X)
  
  best <- NULL
  best_obj <- -Inf
  
  for (s in 1:n_start) {
    set.seed(seed_base + 1000*s + 17*m)
    
    # avoid empty components in init
    z <- sample.int(m, n, replace = TRUE)
    tries <- 0
    while (any(tabulate(z, nbins = m) == 0) && tries < 50) {
      z <- sample.int(m, n, replace = TRUE)
      tries <- tries + 1
    }
    
    tau <- matrix(0, n, m)
    tau[cbind(1:n, z)] <- 1
    
    pi <- colMeans(tau); pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
    rho <- rep(1 / (sd(y) + 1e-12), m)
    
    beta_tilde <- matrix(0, p, m)
    
    pen_w_list <- vector("list", m)
    for (j in 1:m) pen_w_list[[j]] <- rep(1, p)
    
    if (adaptive) {
      # ridge-WLS initializer for beta_tilde to avoid many exact zeros
      beta_init <- matrix(0, p, m)
      for (j in 1:m) {
        beta_init[, j] <- ridge_wls(X, rho[j] * y, w = tau[, j], ridge = ridge_init)
      }
      eps <- 1e-6
      for (j in 1:m) {
        wraw <- 1 / (abs(beta_init[, j])^gamma + eps)
        pen_w_list[[j]] <- pmin(wraw, w_cap)
      }
      beta_tilde <- beta_init
    }
    
    ll_old <- -Inf
    for (iter in 1:max_em) {
      tau <- estep_tau(y, X, beta_tilde, rho, pi)
      
      ms <- mstep_mix_lasso(y, X, tau, beta_tilde, rho, pi,
                            lam = lam,
                            pen_w_list = pen_w_list,
                            max_cd = max_cd, tol_cd = tol_cd,
                            min_pi = min_pi,
                            w_min = w_min,
                            rho_min = rho_min, rho_max = rho_max)
      
      beta_tilde <- ms$beta_tilde
      rho <- ms$rho
      pi <- ms$pi
      
      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
      
      # FIX: save ll before break
      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }
    
    pen <- 0
    for (j in 1:m) pen <- pen + sum(pen_w_list[[j]] * abs(beta_tilde[, j]))
    obj <- ll_old - n * lam * pen
    
    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta_tilde = beta_tilde, rho = rho, pi = pi, ll = ll_old,
                   pen_w_list = pen_w_list, m = m, lam = lam, adaptive = adaptive)
    }
  }
  best
}


# -------------------------
# Fit mixture model for fixed (m, lam) by EM (FIXED ll_old)
# Mix-HP-L / Mix-HP-AL
# -------------------------
fit_mix_HP_L_or_AL <- function(y, X, m, lam,
                               adaptive = FALSE, gamma = 1,
                               max_em = 200, tol_em = 1e-5,
                               max_cd = 200, tol_cd = 1e-6,
                               n_start = 5, seed_base = 1,
                               min_pi = 1e-6,
                               w_min = 5,
                               rho_min = 1e-3, rho_max = 1e3,
                               w_cap = 1e4,
                               ridge_init = 1e-4) {
  n <- length(y); p <- ncol(X)
  
  best <- NULL
  best_obj <- -Inf
  
  for (s in 1:n_start) {
    set.seed(seed_base + 2000*s + 23*m)
    
    # avoid empty components in init
    z <- sample.int(m, n, replace = TRUE)
    tries <- 0
    while (any(tabulate(z, nbins = m) == 0) && tries < 50) {
      z <- sample.int(m, n, replace = TRUE)
      tries <- tries + 1
    }
    
    tau <- matrix(0, n, m)
    tau[cbind(1:n, z)] <- 1
    
    pi <- colMeans(tau); pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
    rho <- rep(1 / (sd(y) + 1e-12), m)
    
    beta0 <- rep(0, p)
    Bfree <- matrix(0, p, m - 1)
    
    pen_w0 <- rep(1, p)
    pen_wB <- matrix(1, p, m)
    
    if (adaptive) {
      # ridge-WLS initializer for full beta_tilde (phi), then decompose to beta0 + B
      beta_init <- matrix(0, p, m)
      for (j in 1:m) {
        beta_init[, j] <- ridge_wls(X, rho[j] * y, w = tau[, j], ridge = ridge_init)
      }
      beta0_init <- rowMeans(beta_init)
      Bfull_init <- sweep(beta_init, 1, beta0_init, "-")  # sums to 0 (approx) by row
      
      # enforce exact sum-to-zero (numeric)
      Bfull_init <- sweep(Bfull_init, 1, rowMeans(Bfull_init), "-")
      
      Bfree_init <- Bfull_init[, 1:(m - 1), drop = FALSE]
      
      eps <- 1e-6
      pen_w0 <- pmin(1 / (abs(beta0_init)^gamma + eps), w_cap)
      pen_wB <- pmin(1 / (abs(Bfull_init)^gamma + eps), w_cap)
      
      beta0 <- beta0_init
      Bfree <- Bfree_init
    }
    
    ll_old <- -Inf
    for (iter in 1:max_em) {
      Bfull <- cbind(Bfree, -rowSums(Bfree))
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
      
      tau <- estep_tau(y, X, beta_tilde, rho, pi)
      
      ms <- mstep_mix_hp(y, X, tau, beta0, Bfree, rho, pi,
                         lam = lam,
                         pen_w0 = pen_w0,
                         pen_wB = pen_wB,
                         max_cd = max_cd, tol_cd = tol_cd,
                         min_pi = min_pi,
                         w_min = w_min,
                         rho_min = rho_min, rho_max = rho_max,
                         rebuild_every = 5)
      
      beta0 <- ms$beta0
      Bfree <- ms$Bfree
      rho <- ms$rho
      pi <- ms$pi
      
      Bfull <- ms$Bfull
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
      
      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
      
      # FIX: save ll before break
      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }
    
    Bfull <- cbind(Bfree, -rowSums(Bfree))
    pen <- sum(pen_w0 * abs(beta0)) + sum(pen_wB * abs(Bfull))
    obj <- ll_old - n * lam * pen
    
    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                   rho = rho, pi = pi, ll = ll_old,
                   pen_w0 = pen_w0, pen_wB = pen_wB,
                   m = m, lam = lam, adaptive = adaptive)
    }
  }
  best
}


# -------------------------
# BIC helper
# -------------------------
bic_mix <- function(ll, df, n) {
  -2 * ll + df * log(n)
}


# -------------------------
# Lambda grid (m-specific)
# baseline: KKT at beta=0 uses sum_i tau_ij x_i ytilde_i
# with tau roughly 1/m => lam_max roughly (1/m) of single-regression lam_max
# -------------------------
lambda_grid_m <- function(y, X, m, nlambda = 25, ratio = 0.02) {
  n <- length(y)
  rho0 <- 1 / (sd(y) + 1e-12)
  yt <- rho0 * y
  lam_max0 <- max(abs(colSums(X * yt))) / (n + 1e-12)
  lam_max <- lam_max0 / max(m, 1)
  lam_min <- lam_max * ratio
  exp(seq(log(lam_max), log(lam_min), length.out = nlambda))
}


# -------------------------
# Fit one method over m in {2,3,4} and lambda grid; pick best by BIC
# (FIXED HP df uses Bfree)
# -------------------------
fit_by_bic <- function(y, X, method,
                       m_set = 2:4,
                       nlambda = 25, lam_ratio = 0.02,
                       n_start = 5,
                       seed_base = 1) {
  n <- length(y); p <- ncol(X)
  
  best_all <- NULL
  best_bic <- Inf
  
  for (m in m_set) {
    lams <- lambda_grid_m(y, X, m, nlambda = nlambda, ratio = lam_ratio)
    
    for (lam in lams) {
      
      if (method == "Mix-L") {
        fit <- fit_mix_L_or_AL(y, X, m, lam, adaptive = FALSE,
                               n_start = n_start, seed_base = seed_base)
        beta_tilde <- fit$beta_tilde
        ll <- fit$ll
        df <- count_nz(beta_tilde) + (m - 1) + m
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
        
      } else if (method == "Mix-AL") {
        fit <- fit_mix_L_or_AL(y, X, m, lam, adaptive = TRUE,
                               n_start = n_start, seed_base = seed_base)
        beta_tilde <- fit$beta_tilde
        ll <- fit$ll
        df <- count_nz(beta_tilde) + (m - 1) + m
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
        
      } else if (method == "Mix-HP-L") {
        fit <- fit_mix_HP_L_or_AL(y, X, m, lam, adaptive = FALSE,
                                  n_start = n_start, seed_base = seed_base)
        beta0 <- fit$beta0
        Bfree <- fit$Bfree
        Bfull <- fit$Bfull
        beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
        beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
        ll <- fit$ll
        
        # FIX: df uses Bfree (independent parameters)
        df <- count_nz(beta0) + count_nz(Bfree) + (m - 1) + m
        
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
        
      } else if (method == "Mix-HP-AL") {
        fit <- fit_mix_HP_L_or_AL(y, X, m, lam, adaptive = TRUE,
                                  n_start = n_start, seed_base = seed_base)
        beta0 <- fit$beta0
        Bfree <- fit$Bfree
        Bfull <- fit$Bfull
        beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
        beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
        ll <- fit$ll
        
        # FIX: df uses Bfree (independent parameters)
        df <- count_nz(beta0) + count_nz(Bfree) + (m - 1) + m
        
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
        
      } else {
        stop("Unknown method")
      }
    }
  }
  best_all
}


# -------------------------
# Component matching for MSE (m<=4)
# -------------------------
perms <- function(v) {
  if (length(v) == 1) return(list(v))
  out <- list()
  for (i in seq_along(v)) {
    rest <- v[-i]
    ps <- perms(rest)
    for (p in ps) out[[length(out) + 1]] <- c(v[i], p)
  }
  out
}

match_to_true <- function(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true) {
  p <- nrow(B_true)
  m_true <- ncol(B_true)
  m_hat <- ncol(B_hat)
  
  best_sse <- Inf
  best <- NULL
  
  if (m_hat >= m_true) {
    idx_list <- combn(m_hat, m_true, simplify = FALSE)
    pset <- perms(1:m_true)
    for (ss in idx_list) {
      for (pp in pset) {
        idx <- ss[pp]
        Bh <- B_hat[, idx, drop = FALSE]
        sse <- sum((Bh - B_true)^2)
        if (sse < best_sse) {
          best_sse <- sse
          best <- list(B = Bh, pi = pi_hat[idx], s2 = s2_hat[idx], m_hat = m_hat)
        }
      }
    }
    return(best)
  }
  
  true_subs <- combn(m_true, m_hat, simplify = FALSE)
  pset <- perms(1:m_hat)
  for (ts in true_subs) {
    Bt_sub <- B_true[, ts, drop = FALSE]
    for (pp in pset) {
      Bh <- B_hat[, pp, drop = FALSE]
      sse <- sum((Bh - Bt_sub)^2)
      miss <- setdiff(1:m_true, ts)
      if (length(miss) > 0) sse <- sse + sum((B_true[, miss, drop = FALSE])^2)
      if (sse < best_sse) {
        Bpad <- matrix(0, p, m_true)
        pipad <- rep(0, m_true)
        s2pad <- rep(0, m_true)
        Bpad[, ts] <- Bh
        pipad[ts] <- pi_hat[pp]
        s2pad[ts] <- s2_hat[pp]
        best_sse <- sse
        best <- list(B = Bpad, pi = pipad, s2 = s2pad, m_hat = m_hat)
      }
    }
  }
  best
}


# -------------------------
# Metrics
# - MSEs computed after matching (Bh)
# - FPR/TPR computed using PRE-matching B_hat (recommended)
# -------------------------
compute_metrics <- function(fit, truth, tol_sel = 1e-6) {
  B_true <- truth$b_true
  pi_true <- truth$pi_true
  s2_true <- truth$sigma2_true
  
  beta_tilde <- fit$beta_tilde
  rho <- fit$rho
  pi_hat <- fit$pi
  
  s2_hat <- (1 / (rho + 1e-12))^2
  
  # PRE-matching coefficients (unscaled b_hat)
  B_hat <- sweep(beta_tilde, 2, rho, "/")
  
  matched <- match_to_true(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true)
  Bh <- matched$B; pih <- matched$pi; s2h <- matched$s2
  
  mse_b  <- mean((Bh - B_true)^2)
  mse_pi <- mean((pih - pi_true)^2)
  mse_s2 <- mean((s2h - s2_true)^2)
  
  p <- nrow(B_true)
  S_R <- truth$S_relevant
  S_C <- truth$S_common
  
  # relevant selection: any component has nonzero b_hat
  sel_R <- which(apply(abs(B_hat), 1, function(v) any(v > tol_sel)))
  
  # heterogeneity selection:
  # paper focuses on Definition 2 (scaled coefficients), so use beta_tilde variation across components
  if (!is.null(fit$method) && fit$method %in% c("Mix-HP-L", "Mix-HP-AL")) {
    Bfull <- fit$Bfull
    sel_H <- which(apply(abs(Bfull), 1, function(v) any(v > tol_sel)))
  } else {
    sel_H <- which(apply(beta_tilde, 1, function(v) sd(v) > tol_sel))
  }
  
  FPR <- length(intersect(sel_R, setdiff(1:p, S_R))) / (p - length(S_R))
  TPR <- length(intersect(sel_R, S_R)) / length(S_R)
  FHR <- length(intersect(sel_H, S_C)) / length(S_C)
  
  list(
    mse_b = mse_b,
    mse_pi = mse_pi,
    mse_sigma2 = mse_s2,
    FPR = FPR, TPR = TPR, FHR = FHR,
    m_hat = matched$m_hat
  )
}


# -------------------------
# One replicate: fit 4 methods and evaluate
# -------------------------
one_rep <- function(rep_id, n, p, snr, correlated,
                    nlambda = 25, lam_ratio = 0.02,
                    n_start = 5, seed_base = 2026) {
  
  dat <- simulate_dataset(n = n, p = p, snr = snr, correlated = correlated,
                          seed = seed_base + rep_id)
  y <- dat$y; X <- dat$X
  
  methods <- c("Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL")
  out <- vector("list", length(methods))
  
  for (i in seq_along(methods)) {
    meth <- methods[i]
    fit <- fit_by_bic(y, X, meth,
                      m_set = 2:4,
                      nlambda = nlambda,
                      lam_ratio = lam_ratio,
                      n_start = n_start,
                      seed_base = seed_base + 999*rep_id)
    
    fit$method <- meth
    fit$y <- y; fit$X <- X
    
    met <- compute_metrics(fit, dat, tol_sel = 1e-6)
    
    out[[i]] <- data.frame(
      rep = rep_id, method = meth,
      snr = snr, p = p, correlated = correlated,
      mse_b = met$mse_b,
      mse_sigma2 = met$mse_sigma2,
      mse_pi = met$mse_pi,
      FPR = met$FPR, TPR = met$TPR, FHR = met$FHR,
      m_hat = met$m_hat,
      stringsAsFactors = FALSE
    )
  }
  
  do.call(rbind, out)
}


# -------------------------
# Run simulation grid
# -------------------------
run_sim <- function(R = 20, n = 200, p = 60,
                    snr_grid = c(200, 100, 50, 25, 12.5),
                    correlated = FALSE,
                    nlambda = 25, lam_ratio = 0.02,
                    n_start = 5, seed_base = 2026) {
  
  res_all <- list()
  idx <- 1
  
  for (snr in snr_grid) {
    for (r in 1:R) {
      cat("SNR=", snr, " rep=", r, " / ", R, "\n")
      res_all[[idx]] <- one_rep(r + 10000*snr, n, p, snr, correlated,
                                nlambda = nlambda, lam_ratio = lam_ratio,
                                n_start = n_start, seed_base = seed_base)
      idx <- idx + 1
    }
  }
  
  do.call(rbind, res_all)
}


# -------------------------
# Summary like Table (paper reports MSE*100 and rates in %)
# -------------------------
summarize_res <- function(res) {
  methods <- unique(res$method)
  out <- data.frame()
  
  for (snr in sort(unique(res$snr))) {
    for (meth in methods) {
      sub <- res[res$snr == snr & res$method == meth, ]
      if (nrow(sub) == 0) next
      out <- rbind(out, data.frame(
        snr = snr,
        method = meth,
        mse_b_100 = 100 * mean(sub$mse_b, na.rm = TRUE),
        mse_s2_100 = 100 * mean(sub$mse_sigma2, na.rm = TRUE),
        mse_pi_100 = 100 * mean(sub$mse_pi, na.rm = TRUE),
        FPR_pct = 100 * mean(sub$FPR, na.rm = TRUE),
        FHR_pct = 100 * mean(sub$FHR, na.rm = TRUE),
        TPR_pct = 100 * mean(sub$TPR, na.rm = TRUE),
        m_hat_mode = {
          mh <- sub$m_hat[is.finite(sub$m_hat)]
          if (length(mh) == 0) NA else as.integer(names(sort(table(mh), decreasing = TRUE)[1]))
        },
        stringsAsFactors = FALSE
      ))
    }
  }
  out
}


# -------------------------
# Figure-1 style: log(MSE) boxplot for coefficients (base R)
# -------------------------
plot_fig1_like <- function(res, snr_value) {
  sub <- res[res$snr == snr_value, ]
  sub <- sub[is.finite(sub$mse_b), ]
  if (nrow(sub) == 0) return(invisible(NULL))
  boxplot(log(sub$mse_b) ~ sub$method,
          main = paste0("log(MSE) of coefficients, SNR=", snr_value),
          xlab = "", ylab = "log(MSE)")
}


# -------------------------
# Paper Table style summary (mean (SE))
# -------------------------
summarize_paper_style <- function(res) {
  methods_order <- c("Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL")
  out <- data.frame()
  
  for (snr in sort(unique(res$snr), decreasing = TRUE)) {
    for (meth in methods_order) {
      sub <- res[res$snr == snr & res$method == meth, ]
      if (nrow(sub) == 0) next
      
      n_rep <- nrow(sub)
      
      format_stat <- function(values, scale = 1) {
        val <- values * scale
        avg <- mean(val, na.rm = TRUE)
        se  <- sd(val, na.rm = TRUE) / sqrt(n_rep)
        sprintf("%.2f (%.2f)", avg, se)
      }
      
      format_rate <- function(values) {
        sprintf("%.1f", mean(values, na.rm = TRUE) * 100)
      }
      
      out <- rbind(out, data.frame(
        SNR = snr,
        Method = meth,
        b = format_stat(sub$mse_b, 100),
        sigma2 = format_stat(sub$mse_sigma2, 100),
        pi = format_stat(sub$mse_pi, 100),
        FPR = format_rate(sub$FPR),
        FHR = format_rate(sub$FHR),
        TPR = format_rate(sub$TPR),
        m_hat = {
          mh <- sub$m_hat[is.finite(sub$m_hat)]
          if (length(mh) == 0) NA else as.integer(names(sort(table(mh), decreasing = TRUE)[1]))
        },
        stringsAsFactors = FALSE
      ))
    }
  }
  
  colnames(out) <- c("SNR", "Method", "MSE_b(SE)", "MSE_sigma2(SE)", "MSE_pi(SE)", "FPR(%)", "FHR(%)", "TPR(%)", "m_mode")
  out
}


# ============================================================
# Example run (sanity check):
# ============================================================
set.seed(1)
res <- run_sim(R = 5, n = 200, p = 60,
               snr_grid = c(50),
               correlated = FALSE,
               nlambda = 10, lam_ratio = 0.05,
               n_start = 2, seed_base = 2026)

print(summarize_res(res))
plot_fig1_like(res, 50)

final_table <- summarize_paper_style(res)
print(final_table, row.names = FALSE)
