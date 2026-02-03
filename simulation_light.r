# ============================================================
# Li et al. (Biometrics) simulation reproduction (CORRECTED + TRACE)
#
# 목적
# - 혼합 회귀(mixture regression) 시뮬레이션을 생성하고,
#   4가지 추정법(Mix-L, Mix-AL, Mix-HP-L, Mix-HP-AL)을 구현/비교한다.
# - m ∈ {2,3,4}, lambda grid에 대해 BIC 최소값으로 모델 선택.
#
# 핵심 파라미터화(scaled form)
# - 원모형: y_i | z_i=j ~ N( X_i^T b_j , sigma_j^2 )
# - 코드에서는 rho_j = 1/sigma_j, beta_tilde_j = rho_j * b_j 를 사용.
#   r_ij = rho_j*y_i - X_i^T beta_tilde_j 로 쓰면,
#   log-likelihood에 log(rho_j) 항이 자연스럽게 들어가고 EM/M-step이 단순해진다.
#
# HP(Heterogeneity Pursuit) 분해
# - beta_tilde_j = beta0 (common) + B_j (hetero),  sum_j B_j = 0 제약.
# - 구현은 Bfree(첫 m-1개)만 최적화하고 마지막은 -rowSums(Bfree)로 강제.
#
# 평가
# - label switching 때문에 component 순서를 permutation 매칭한 뒤
#   MSE(b), MSE(pi), MSE(sigma^2), FPR/TPR/FHR 등을 계산한다.
# ============================================================

options(stringsAsFactors = FALSE)

# =========================
# LIGHT MODE CONFIG
# =========================
# - 코드 구조 + 정상 작동 확인용 경량 설정
# - 핵심 3개: R=2, nlambda=30, max_em=100
LIGHT_MODE <- TRUE

CFG <- list(
  R        = 2,      # 반복수(Rep)
  n        = 200,
  p        = 60,
  snr      = 50,
  nlambda  = 30,     # lambda grid 개수
  max_em   = 100,    # EM 최대 반복
  tol_em   = 1e-5,
  pilot_em = 30,
  n_start  = 10,     # random start 개수(속도 줄이려면 3~5도 가능)
  max_cd   = 1000,   # CD 최대 반복(속도 줄이려면 300~500도 가능)
  tol_cd   = 1e-7,
  min_cd   = 10
)

if (LIGHT_MODE) {
  CFG$R       <- 2
  CFG$nlambda <- 30
  CFG$max_em  <- 100
  
  # 아래 2개는 “선택적” 속도 옵션
  # CFG$n_start <- 3
  # CFG$max_cd  <- 400
}

# -------------------------
# Utilities (수치/로그 안정성용)
# -------------------------

# safe_seed(x): seed 입력을 안전한 int 범위로 변환(재현성 + 오류 방지)
safe_seed <- function(x) {
  m <- .Machine$integer.max
  s <- suppressWarnings(as.numeric(x))
  if (length(s) != 1 || is.na(s) || !is.finite(s)) return(1L)
  s <- s %% m
  if (s <= 0) s <- s + (m - 1)
  as.integer(s)
}

# soft_thresh(z, g): L1 penalty의 soft-thresholding
soft_thresh <- function(z, g) {
  if (z >  g) return(z - g)
  if (z < -g) return(z + g)
  0
}

# logsumexp(v): log(sum(exp(v))) 안정 계산(underflow/overflow 방지)
logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

# clamp: adaptive weight 폭주 방지용
clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

# 출력 포맷
fmt_num <- function(x, d = 6) {
  if (length(x) == 0 || all(is.na(x))) return("NA")
  formatC(x, digits = d, format = "f")
}
now_str <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

# -------------------------
# Covariance matrices (X 생성용)
# -------------------------
# correlated=TRUE면 Toeplitz(0.5^{|i-j|}), 기본은 독립(diag)
cov_mat <- function(p, correlated = FALSE) {
  if (!correlated) return(diag(p))
  toeplitz(0.5^(0:(p - 1)))
}

# -------------------------
# Adaptive weights (Adaptive Lasso)
# -------------------------
# - w_k = 1/(|beta_k|^gamma + eps)
# - "정규화하지 않음": penalty 크기/스케일을 임의로 바꾸지 않기 위함
make_adaptive_weights <- function(beta_mat, gamma = 1,
                                  eps = 1e-4, w_min = 1e-2, w_max = 1e4) {
  w <- 1 / (abs(beta_mat)^gamma + eps)
  w <- clamp(w, w_min, w_max)
  w
}

# -------------------------
# SNR <-> delta mapping
# -------------------------
# 논문 세팅을 따라 noise scale(delta)을 SNR과 연결
delta_from_snr <- function(snr) 25 / snr

# -------------------------
# True scaled betas (신호 구조)
# -------------------------
# - relevant: 1:10
# - common:   1:7
# - hetero:   8:10 (component별로 상이)
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
  
  list(beta0 = beta0, beta_het = cbind(beta1, beta2, beta3))
}

# ============================================================
# Data generator (시뮬레이션 데이터 생성)
# ============================================================
# simulate_dataset()
# - X ~ N(0, Sigma)
# - m_true=3, pi_true=(1/3,1/3,1/3)
# - sigma^2_true = delta*(0.1,0.1,0.4)
# - scaled beta_tilde_true = beta0 + beta_het[,j]
# - 원모형 계수 b_true = beta_tilde_true / rho_true
# - y_i = X_i^T b_{z_i} + N(0, sigma_{z_i}^2)
simulate_dataset <- function(n = 200, p = 60, snr = 50, correlated = FALSE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  delta <- delta_from_snr(snr)
  Sigma <- cov_mat(p, correlated)
  
  X <- matrix(rnorm(n * p), n, p)
  U <- chol(Sigma)
  X <- X %*% U
  
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
  
  # 핵심: i별로 자기 component 계수(b_true[, z_i])와 X_i 내적을 계산
  mu <- rowSums(X * t(b_true[, z, drop = FALSE]))
  y  <- rnorm(n, mean = mu, sd = sigma_true[z])
  
  list(
    y = y, X = X, z = z,
    delta = delta, snr = snr, correlated = correlated,
    b_true = b_true,
    beta_tilde_true = beta_tilde_true,
    pi_true = pi_true,
    sigma2_true = sigma2_true,
    rho_true = rho_true,
    S_relevant = 1:10, S_common = 1:7, S_hetero = 8:10
  )
}

# ============================================================
# Log-likelihood & E-step (scaled form)
# ============================================================
# loglik_mix_scaled:
# - r_ij = rho_j*y_i - X_i^T beta_tilde_j
# - log f(y_i|j) ∝ log(rho_j) - 0.5 r_ij^2 (상수항 포함)
loglik_mix_scaled <- function(y, X, beta_tilde_hat, rho_hat, pi_hat) {
  n <- length(y); m <- length(pi_hat)
  ll <- 0
  for (i in 1:n) {
    lvec <- rep(NA_real_, m)
    for (j in 1:m) {
      mu <- sum(X[i, ] * beta_tilde_hat[, j])
      r  <- rho_hat[j] * y[i] - mu
      lvec[j] <- log(pi_hat[j] + 1e-300) + log(rho_hat[j] + 1e-300) -
        0.5 * r * r - 0.5 * log(2*pi)
    }
    ll <- ll + logsumexp(lvec)
  }
  ll
}

# estep_tau:
# - 책임도 tau_{ij} = P(z_i=j | y_i, X_i) 계산
# - log-space에서 logsumexp로 normalize
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

# ============================================================
# rho update (sigma update 대체)
# ============================================================
# update_rho_quad:
# - M-step에서 rho_j=1/sigma_j를 가중 구조로 업데이트(수치 안정)
# - rho_floor로 0에 붙는 것 방지
update_rho_quad <- function(y, mu, w, rho_floor = 1e-8) {
  A <- sum(w) + 1e-12
  C <- sum(w * y^2) + 1e-12
  B <- sum(w * y * mu)
  disc <- B^2 + 4 * A * C
  rho  <- (B + sqrt(disc)) / (2 * C)
  max(rho, rho_floor)
}

# ============================================================
# Weighted Ridge WLS (초기화 / pilot EM)
# ============================================================
# ridge_wls:
# - (X^T W X + alpha I)^{-1} X^T W y
# - solve 실패 시 qr.solve fallback
ridge_wls <- function(X, y, w, alpha) {
  XtW <- t(X) * w
  A <- XtW %*% X
  diag(A) <- diag(A) + alpha
  b <- XtW %*% y
  out <- tryCatch(as.vector(solve(A, b)), error = function(e) NULL)
  if (is.null(out)) out <- as.vector(qr.solve(A, b))
  out
}

# init_from_tau:
# - hard/soft assignment(tau)로부터 초기 pi, rho, beta_tilde 생성
# - 각 component별 ridge로 b_j -> residual로 sigma^2 -> rho=1/sqrt(s2)
# - scaled 계수 beta_tilde = b_j * rho_j 로 저장
init_from_tau <- function(y, X, tau,
                          ridge_scale = 1e-3,
                          sigma2_floor = 1e-6,
                          rho_floor = 1e-8,
                          min_pi = 1e-6) {
  n <- length(y); p <- ncol(X); m <- ncol(tau)
  
  pi <- colMeans(tau)
  pi <- pmax(pi, min_pi)
  pi <- pi / sum(pi)
  
  b_init   <- matrix(0, p, m)
  phi_init <- matrix(0, p, m)
  rho_init <- rep(1 / (sd(y) + 1e-12), m)
  
  for (j in 1:m) {
    w <- tau[, j]
    sw <- sum(w)
    if (sw < 5) {
      b_init[,j] <- rnorm(p, 0, 0.01)
      next
    }
    
    xtwx_diag_mean <- mean(colSums((X^2) * w)) + 1e-12
    alpha <- ridge_scale * xtwx_diag_mean
    
    b_j <- ridge_wls(X, y, w, alpha)
    
    r_j <- y - as.vector(X %*% b_j)
    s2j <- sum(w * r_j^2) / (sw + 1e-12)
    s2j <- max(s2j, sigma2_floor)
    
    rho_j <- 1 / sqrt(s2j)
    rho_j <- max(rho_j, rho_floor)
    
    rho_init[j] <- rho_j
    b_init[, j] <- b_j
    phi_init[, j] <- b_j * rho_j
  }
  
  list(pi = pi, rho = rho_init, beta_tilde = phi_init, b = b_init)
}

# ============================================================
# Coordinate Descent solvers
# ============================================================
# lasso_cd_weighted:
# - argmin_b 0.5*sum_i w_i (y_i - x_i^T b)^2 + lam*sum_k pen_w[k]|b_k|
# - 좌표별 soft-thresholding으로 업데이트
lasso_cd_weighted <- function(X, y, w, lam, pen_w = NULL,
                              b_init = NULL,
                              maxit = 500, tol = 1e-8, minit = 10) {
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
    if (it >= minit && max(abs(b - b_old)) < tol) break
  }
  b
}

# update_two_abs_1d:
# - HP에서 |b| + |b+s| 구조(two absolute penalties) 1D 업데이트
# - 후보해들을 열거하여 목적함수 최소값 선택(안정성 우선)
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

# ============================================================
# M-step: Mix-L / Mix-AL (component별 Lasso/Adaptive Lasso)
# ============================================================
# mstep_mix_lasso:
# - pi 업데이트: mean(tau)
# - beta_tilde_j 업데이트: weighted lasso_cd_weighted (yj = rho_j*y 사용)
# - rho 업데이트: update_rho_quad
mstep_mix_lasso <- function(y, X, tau, beta_tilde, rho, pi,
                            lam, pen_w_list,
                            max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                            min_pi = 1e-6, rho_floor = 1e-8) {
  n <- length(y); m <- ncol(tau); p <- ncol(X)
  
  pi_new <- colMeans(tau); pi_new <- pmax(pi_new, min_pi); pi_new <- pi_new / sum(pi_new)
  beta_new <- beta_tilde; rho_new  <- rho
  
  for (j in 1:m) {
    w <- tau[, j]
    yj <- rho_new[j] * y
    beta_new[, j] <- lasso_cd_weighted(
      X, yj, w, lam = n * lam, pen_w = pen_w_list[[j]],
      b_init = beta_new[, j], maxit = max_cd, tol = tol_cd, minit = min_cd
    )
  }
  
  for (j in 1:m) {
    w  <- tau[, j]
    mu <- as.vector(X %*% beta_new[, j])
    rho_new[j] <- update_rho_quad(y, mu, w, rho_floor = rho_floor)
  }
  list(beta_tilde = beta_new, rho = rho_new, pi = pi_new)
}

# ============================================================
# M-step: Mix-HP (HP: common + hetero, sum-to-zero 제약)
# ============================================================
# mstep_mix_hp:
# - beta_tilde_j = beta0 + B_j, sum_j B_j = 0
# - Bfree(첫 m-1개)만 업데이트하고 B_m = -rowSums(Bfree)
# - CD로 beta0(공통) 먼저 업데이트 후 Bfree(이질) 업데이트
# - rebuild_every 주기마다 잔차를 재구축(드리프트 방지)
mstep_mix_hp <- function(y, X, tau, beta0, Bfree, rho, pi,
                         lam, pen_w0, pen_wB,
                         max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                         rebuild_every = 5, min_pi = 1e-6, rho_floor = 1e-8,
                         verbose = FALSE, trace_every = 50, tag = "CD-HP") {
  n <- length(y); m <- ncol(tau); p <- ncol(X)
  
  pi_new <- colMeans(tau); pi_new <- pmax(pi_new, min_pi); pi_new <- pi_new / sum(pi_new)
  rho_new <- rho; beta0_new <- beta0; Bfree_new <- Bfree
  
  make_Bfull <- function(Bf) cbind(Bf, -rowSums(Bf))
  Xcols <- lapply(1:p, function(k) X[, k])
  
  build_residuals <- function(beta0v, Bfullv, rhov) {
    r <- matrix(0, n, m)
    for (j in 1:m) {
      bt <- beta0v + Bfullv[, j]
      r[, j] <- rhov[j] * y - as.vector(X %*% bt)
    }
    r
  }
  
  Bfull <- make_Bfull(Bfree_new)
  rmat <- build_residuals(beta0_new, Bfull, rho_new)
  
  for (it in 1:max_cd) {
    maxchg <- 0
    
    # ---- Update beta0 (common) ----
    for (k in 1:p) {
      xk <- Xcols[[k]]
      a <- sum((xk^2) * rowSums(tau))
      if (a < 1e-12) next
      
      cval <- 0
      for (j in 1:m) cval <- cval + sum(tau[, j] * xk * (rmat[, j] + xk * beta0_new[k]))
      
      bnew <- soft_thresh(cval, n * lam * pen_w0[k]) / a
      d <- bnew - beta0_new[k]
      if (abs(d) > 0) {
        beta0_new[k] <- bnew
        maxchg <- max(maxchg, abs(d))
        for (j in 1:m) rmat[, j] <- rmat[, j] - xk * d
      }
    }
    
    # ---- Update Bfree (hetero) ----
    # 마지막 component(m)은 B_m = -sum_{l=1}^{m-1} B_l 로 묶여 있음
    for (l in 1:(m - 1)) {
      for (k in 1:p) {
        xk <- Xcols[[k]]
        s_minus <- if (m - 1 == 1) 0 else sum(Bfree_new[k, -l])
        bcur <- Bfree_new[k, l]
        
        a <- sum((tau[, l] + tau[, m]) * (xk^2))
        if (a < 1e-12) next
        
        partial_l <- rmat[, l] + xk * bcur
        partial_m <- rmat[, m] - xk * bcur
        
        # CRITICAL: sign 구조
        # l의 기여는 +, 마지막 component(m)의 기여는 - 로 들어가야
        cval <- sum(tau[, l] * xk * partial_l - tau[, m] * xk * partial_m)
        
        lam1 <- n * lam * pen_wB[k, l]
        lam2 <- n * lam * pen_wB[k, m]
        
        bnew <- update_two_abs_1d(a = a, c = cval, s = s_minus, lam1 = lam1, lam2 = lam2)
        d <- bnew - bcur
        if (abs(d) > 0) {
          Bfree_new[k, l] <- bnew
          maxchg <- max(maxchg, abs(d))
          rmat[, l] <- rmat[, l] - xk * d
          rmat[, m] <- rmat[, m] + xk * d
        }
      }
    }
    
    if (rebuild_every > 0 && (it %% rebuild_every) == 0) {
      Bfull <- make_Bfull(Bfree_new)
      rmat <- build_residuals(beta0_new, Bfull, rho_new)
    }
    
    if (verbose && (it %% trace_every == 0)) {
      cat(sprintf("[%s] cd_iter=%d maxchg=%s\n", tag, it, fmt_num(maxchg, 6)))
    }
    
    if (it >= min_cd && maxchg < tol_cd) break
  }
  
  # rho 업데이트(최종 beta0/Bfull로 mu 계산)
  Bfull <- make_Bfull(Bfree_new)
  for (j in 1:m) {
    w  <- tau[, j]
    mu <- as.vector(X %*% (beta0_new + Bfull[, j]))
    rho_new[j] <- update_rho_quad(y, mu, w, rho_floor = rho_floor)
  }
  
  list(beta0 = beta0_new, Bfull = Bfull, Bfree = Bfree_new, rho = rho_new, pi = pi_new)
}

# ============================================================
# Pilot Ridge EM (Adaptive weights 생성용 기반)
# ============================================================
# pilot_ridge_em_mix:
# - penalty 없이 ridge 기반 EM을 짧게 돌려 안정적인 beta_tilde/rho/pi를 만든다
# - Adaptive Lasso 가중치 계산에 사용
pilot_ridge_em_mix <- function(y, X, m, n_start = 1, seed_base = 1,
                               max_em = 30, ridge_scale = 1e-3,
                               min_pi = 1e-6, rho_floor = 1e-8,
                               verbose = FALSE, trace_every = 10, tag = "PILOT") {
  n <- length(y); p <- ncol(X)
  best <- NULL; best_ll <- -Inf
  
  for (s in 1:n_start) {
    set.seed(safe_seed(seed_base + 777*s + 31*m))
    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m); tau[cbind(1:n, z)] <- 1
    
    ini <- init_from_tau(y, X, tau, ridge_scale = ridge_scale,
                         rho_floor = rho_floor, min_pi = min_pi)
    pi <- ini$pi; rho <- ini$rho; beta_tilde <- ini$beta_tilde
    
    ll_old <- -Inf
    for (iter in 1:max_em) {
      tau <- estep_tau(y, X, beta_tilde, rho, pi)
      pi <- colMeans(tau); pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
      
      for (j in 1:m) {
        w <- tau[, j]
        yj <- rho[j] * y
        xtwx_diag_mean <- mean(colSums((X^2) * w)) + 1e-12
        alpha <- ridge_scale * xtwx_diag_mean
        beta_tilde[, j] <- ridge_wls(X, yj, w, alpha)
      }
      for (j in 1:m) {
        w  <- tau[, j]
        mu <- as.vector(X %*% beta_tilde[, j])
        rho[j] <- update_rho_quad(y, mu, w, rho_floor = rho_floor)
      }
      
      if (verbose && (iter %% trace_every == 0)) {
        ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
        cat(sprintf("[%s] iter=%d ll=%s dll=%s\n", tag, iter, fmt_num(ll, 6), fmt_num(ll-ll_old, 6)))
        ll_old <- ll
      }
    }
    
    ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
    if (ll > best_ll) {
      best_ll <- ll
      best <- list(beta_tilde = beta_tilde, rho = rho, pi = pi, ll = ll)
    }
  }
  best
}

# ============================================================
# Fit Mix-L / Mix-AL (EM + CD + TRACE)
# ============================================================
# fit_mix_L_or_AL:
# - 여러 random start 중 penalized objective 최대 해 선택
# - Mix-AL은 pilot ridge EM으로 adaptive weights를 만든 뒤 고정하고 EM 수행
fit_mix_L_or_AL <- function(y, X, m, lam, adaptive = FALSE, gamma = 1,
                            pilot_em = 30, pilot_ridge_scale = 1e-3,
                            w_eps = 1e-4, w_min = 1e-2, w_max = 1e4,
                            max_em = 200, tol_em = 1e-5,
                            max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                            n_start = 5, seed_base = 1,
                            min_pi = 1e-6, rho_floor = 1e-8,
                            verbose = FALSE, trace_em_every = 10, trace_tag = "EM-MIX") {
  n <- length(y); p <- ncol(X)
  best <- NULL; best_obj <- -Inf
  
  for (s in 1:n_start) {
    set.seed(safe_seed(seed_base + 2000*s + 23*m))
    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m); tau[cbind(1:n, z)] <- 1
    
    ini <- init_from_tau(y, X, tau, ridge_scale = pilot_ridge_scale,
                         rho_floor = rho_floor, min_pi = min_pi)
    pi <- ini$pi; rho <- ini$rho; beta_tilde <- ini$beta_tilde
    
    pen_w_list <- vector("list", m)
    for (j in 1:m) pen_w_list[[j]] <- rep(1, p)
    
    if (adaptive) {
      # pilot로 beta_tilde 안정화 후 adaptive weights 계산
      pil <- pilot_ridge_em_mix(y, X, m, n_start = 1, seed_base = seed_base + s*1000,
                                max_em = pilot_em, ridge_scale = pilot_ridge_scale,
                                min_pi = min_pi, rho_floor = rho_floor,
                                verbose = FALSE)
      beta_init <- pil$beta_tilde
      rho <- pil$rho; pi <- pil$pi; beta_tilde <- beta_init
      
      for (j in 1:m) {
        pen_w_list[[j]] <- as.vector(make_adaptive_weights(
          beta_init[, j], gamma = gamma, eps = w_eps, w_min = w_min, w_max = w_max
        ))
      }
    }
    
    ll_old <- -Inf
    em_iter_done <- 0
    
    for (iter in 1:max_em) {
      tau <- estep_tau(y, X, beta_tilde, rho, pi)
      
      ms <- mstep_mix_lasso(y, X, tau, beta_tilde, rho, pi, lam, pen_w_list,
                            max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                            min_pi = min_pi, rho_floor = rho_floor)
      
      beta_tilde <- ms$beta_tilde; rho <- ms$rho; pi <- ms$pi
      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
      em_iter_done <- iter
      
      if (verbose && (iter %% trace_em_every == 0)) {
        cat(sprintf("[%s] start=%d iter=%d/%d ll=%s dll=%s min(pi)=%s min(rho)=%s\n",
                    trace_tag, s, iter, max_em, fmt_num(ll, 6), fmt_num(ll-ll_old, 6),
                    fmt_num(min(pi), 6), fmt_num(min(rho), 6)))
      }
      
      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }
    
    # penalized objective (선택 기준)
    pen <- 0
    for (j in 1:m) pen <- pen + sum(pen_w_list[[j]] * abs(beta_tilde[, j]))
    obj <- ll_old - n * lam * pen
    
    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta_tilde = beta_tilde, rho = rho, pi = pi, ll = ll_old,
                   pen_w_list = pen_w_list, m = m, lam = lam, adaptive = adaptive,
                   em_iter = em_iter_done)
    }
  }
  best
}

# ============================================================
# Fit Mix-HP-L / Mix-HP-AL (EM + HP-CD + TRACE)
# ============================================================
# fit_mix_HP_L_or_AL:
# - beta_tilde를 beta0 + B로 분해(sum-to-zero)
# - Adaptive인 경우 Mix-L pilot로 beta_pilot을 얻어 pen_w0/pen_wB 생성
fit_mix_HP_L_or_AL <- function(y, X, m, lam, adaptive = FALSE, gamma = 1,
                               pilot_em = 30, w_eps = 1e-4, w_min = 1e-2, w_max = 1e4,
                               max_em = 200, tol_em = 1e-5,
                               max_cd = 1000, tol_cd = 1e-8, min_cd = 10,
                               rebuild_every = 5, n_start = 5, seed_base = 1,
                               min_pi = 1e-6, rho_floor = 1e-8,
                               verbose = FALSE, trace_em_every = 10, trace_tag = "EM-HP",
                               cd_verbose = FALSE, cd_trace_every = 50) {
  n <- length(y); p <- ncol(X)
  best <- NULL; best_obj <- -Inf
  pilot_ridge_scale <- 1e-3
  
  for (s in 1:n_start) {
    set.seed(safe_seed(seed_base + 2000*s + 23*m))
    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m); tau[cbind(1:n, z)] <- 1
    
    ini <- init_from_tau(y, X, tau, ridge_scale = pilot_ridge_scale,
                         rho_floor = rho_floor, min_pi = min_pi)
    pi <- ini$pi; rho <- ini$rho
    beta_tilde_init <- ini$beta_tilde
    
    # beta_tilde -> beta0 + B 분해(초기)
    beta0 <- rowMeans(beta_tilde_init)
    Bfull <- sweep(beta_tilde_init, 1, beta0, "-")
    Bfree <- Bfull[, 1:(m - 1), drop = FALSE]
    
    # 기본은 lasso: pen=1
    pen_w0 <- rep(1, p)
    pen_wB <- matrix(1, p, m)
    
    if (adaptive) {
      # Adaptive weight용 pilot: Mix-L을 한 번 돌려 beta_pilot 확보
      pilot_lam <- 0.001
      pil_fit <- fit_mix_L_or_AL(y, X, m, lam = pilot_lam, adaptive = FALSE,
                                 n_start = 1, seed_base = seed_base + s*999,
                                 max_em = pilot_em, verbose = FALSE)
      
      if (!is.null(pil_fit)) {
        beta_pilot <- pil_fit$beta_tilde
        
        rho <- pil_fit$rho
        pi  <- pil_fit$pi
        
        beta0_est <- rowMeans(beta_pilot)
        Bfull_est <- sweep(beta_pilot, 1, beta0_est, "-")
        
        pen_w0 <- make_adaptive_weights(beta0_est, gamma=gamma, eps=w_eps, w_min=w_min, w_max=w_max)
        pen_wB <- make_adaptive_weights(Bfull_est, gamma=gamma, eps=w_eps, w_min=w_min, w_max=w_max)
        
        beta0 <- beta0_est
        Bfree <- Bfull_est[, 1:(m-1), drop=FALSE]
      }
    }
    
    ll_old <- -Inf
    em_iter_done <- 0
    
    for (iter in 1:max_em) {
      # 현재(beta0,Bfree)로 beta_tilde 재구성
      Bfull <- cbind(Bfree, -rowSums(Bfree))
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
      
      tau <- estep_tau(y, X, beta_tilde, rho, pi)
      
      ms <- mstep_mix_hp(
        y, X, tau, beta0, Bfree, rho, pi, lam, pen_w0, pen_wB,
        max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
        rebuild_every = rebuild_every, min_pi = min_pi, rho_floor = rho_floor,
        verbose = cd_verbose, trace_every = cd_trace_every,
        tag = sprintf("CD-HP(m=%d,start=%d)", m, s)
      )
      
      beta0 <- ms$beta0; Bfree <- ms$Bfree; rho <- ms$rho; pi <- ms$pi
      Bfull <- ms$Bfull
      
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
      
      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
      em_iter_done <- iter
      
      if (verbose && (iter %% trace_em_every == 0)) {
        cat(sprintf("[%s] start=%d iter=%d/%d ll=%s dll=%s min(pi)=%s min(rho)=%s\n",
                    trace_tag, s, iter, max_em, fmt_num(ll, 6), fmt_num(ll-ll_old, 6),
                    fmt_num(min(pi), 6), fmt_num(min(rho), 6)))
      }
      
      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }
    
    # penalized objective (선택 기준)
    Bfull <- cbind(Bfree, -rowSums(Bfree))
    pen <- sum(pen_w0 * abs(beta0)) + sum(pen_wB * abs(Bfull))
    obj <- ll_old - n * lam * pen
    
    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                   rho = rho, pi = pi, ll = ll_old,
                   pen_w0 = pen_w0, pen_wB = pen_wB,
                   m = m, lam = lam, adaptive = adaptive,
                   em_iter = em_iter_done)
    }
  }
  best
}

# ============================================================
# BIC / df / lambda grid
# ============================================================
bic_mix <- function(ll, df, n) -2 * ll + df * log(n)

# count_nz: df 근사에서 nonzero 개수 카운트
count_nz <- function(v, tol = 1e-6) sum(abs(v) > tol)

# lambda_grid:
# - lam_max는 대략 "모든 계수가 0"이 되는 수준(근사)
# - lam_min = lam_max * ratio, log-scale grid
lambda_grid <- function(y, X, nlambda = 50, ratio = 1e-4) {
  n <- length(y)
  rho0 <- 1 / (sd(y) + 1e-12)
  yt <- rho0 * y
  lam_max <- max(abs(colSums(X * yt))) / (n + 1e-12)
  lam_min <- lam_max * ratio
  exp(seq(log(lam_max), log(lam_min), length.out = nlambda))
}

# ============================================================
# fit_by_bic: (method, m, lambda) grid search + TRACE
# ============================================================
# - method별로 fit_* 호출
# - df 근사:
#   * Mix-L / Mix-AL: nz(beta_tilde) + (m-1) + m
#   * Mix-HP-L / Mix-HP-AL: nz(beta0) + nz(Bfree) + (m-1) + m
# - TRACE:
#   * GRID-PROG: 진행률/elapsed/bestBIC 출력
#   * GRID-BEST: 전체 best 갱신 시 상세 출력
fit_by_bic <- function(y, X, method, m_set = 2:4,
                       nlambda = 50, lam_ratio = 1e-4,
                       n_start = 5, seed_base = 1,
                       pilot_em = 30, w_eps = 1e-4, w_min = 1e-2, w_max = 1e4,
                       max_em = 200, tol_em = 1e-5,
                       max_cd = 1000, tol_cd = 1e-8, min_cd = 10,
                       min_pi = 1e-6, rho_floor = 1e-8,
                       verbose = FALSE, show_grid_bar = TRUE,
                       grid_trace_every = NULL,
                       grid_print_best = TRUE,
                       grid_print_progress = TRUE,
                       em_verbose = FALSE,
                       em_trace_every = 10) {
  
  lams <- lambda_grid(y, X, nlambda = nlambda, ratio = lam_ratio)
  if (is.null(grid_trace_every)) grid_trace_every <- max(1L, floor(length(lams) / 10))
  
  tol_df <- 1e-8
  
  best_all <- NULL; best_bic <- Inf
  grid_total <- length(m_set) * length(lams)
  grid_id <- 0
  
  pb <- NULL
  if (show_grid_bar) pb <- txtProgressBar(min = 0, max = grid_total, style = 3)
  
  t_grid0 <- proc.time()
  
  for (m in m_set) {
    best_bic_m <- Inf
    best_ll_m  <- -Inf
    best_lam_m <- NA_real_
    
    for (lam in lams) {
      grid_id <- grid_id + 1
      if (!is.null(pb)) setTxtProgressBar(pb, grid_id)
      
      fit <- tryCatch({
        if (method == "Mix-L") {
          fit_mix_L_or_AL(y, X, m, lam, adaptive = FALSE,
                          n_start = n_start, seed_base = seed_base,
                          max_em = max_em, tol_em = tol_em,
                          max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                          min_pi = min_pi, rho_floor = rho_floor,
                          verbose = em_verbose, trace_em_every = em_trace_every,
                          trace_tag = sprintf("EM-%s(m=%d)", method, m))
        } else if (method == "Mix-AL") {
          fit_mix_L_or_AL(y, X, m, lam, adaptive = TRUE,
                          pilot_em = pilot_em,
                          w_eps = w_eps, w_min = w_min, w_max = w_max,
                          n_start = n_start, seed_base = seed_base,
                          max_em = max_em, tol_em = tol_em,
                          max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                          min_pi = min_pi, rho_floor = rho_floor,
                          verbose = em_verbose, trace_em_every = em_trace_every,
                          trace_tag = sprintf("EM-%s(m=%d)", method, m))
        } else if (method == "Mix-HP-L") {
          fit_mix_HP_L_or_AL(y, X, m, lam, adaptive = FALSE,
                             n_start = n_start, seed_base = seed_base,
                             max_em = max_em, tol_em = tol_em,
                             max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                             min_pi = min_pi, rho_floor = rho_floor,
                             verbose = em_verbose, trace_em_every = em_trace_every,
                             trace_tag = sprintf("EM-%s(m=%d)", method, m))
        } else if (method == "Mix-HP-AL") {
          fit_mix_HP_L_or_AL(y, X, m, lam, adaptive = TRUE,
                             pilot_em = pilot_em,
                             w_eps = w_eps, w_min = w_min, w_max = w_max,
                             n_start = n_start, seed_base = seed_base,
                             max_em = max_em, tol_em = tol_em,
                             max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                             min_pi = min_pi, rho_floor = rho_floor,
                             verbose = em_verbose, trace_em_every = em_trace_every,
                             trace_tag = sprintf("EM-%s(m=%d)", method, m))
        } else {
          NULL
        }
      }, error = function(e) NULL)
      
      if (is.null(fit)) next
      
      ll <- fit$ll
      em_it <- if (!is.null(fit$em_iter)) fit$em_iter else NA_integer_
      minpi <- if (!is.null(fit$pi)) min(fit$pi) else NA_real_
      minrho <- if (!is.null(fit$rho)) min(fit$rho) else NA_real_
      
      if (method %in% c("Mix-L", "Mix-AL")) {
        beta_tilde <- fit$beta_tilde
        df <- count_nz(beta_tilde, tol = tol_df) + (m - 1) + m
        bic <- bic_mix(ll, df, length(y))
        
        if (bic < best_bic_m) {
          best_bic_m <- bic; best_ll_m <- ll; best_lam_m <- lam
        }
        
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta_tilde = beta_tilde, rho = fit$rho, pi = fit$pi,
                           ll = ll, bic = bic, df = df, em_iter = em_it)
          if (grid_print_best) {
            cat(sprintf("\n[%s][GRID-BEST] %s | %d/%d | m=%d lam=%s | em_iter=%s ll=%s df=%d bic=%s | min(pi)=%s min(rho)=%s\n",
                        now_str(), method, grid_id, grid_total, m, fmt_num(lam, 6),
                        as.character(em_it), fmt_num(ll, 6), df, fmt_num(bic, 4),
                        fmt_num(minpi, 6), fmt_num(minrho, 6)))
          }
        }
        
      } else {
        beta0 <- fit$beta0; Bfree <- fit$Bfree; Bfull <- fit$Bfull
        beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
        beta_tilde <- matrix(beta_tilde, nrow = ncol(X), ncol = m)
        
        df <- count_nz(beta0, tol = tol_df) + count_nz(Bfree, tol = tol_df) + (m - 1) + m
        bic <- bic_mix(ll, df, length(y))
        
        if (bic < best_bic_m) {
          best_bic_m <- bic; best_ll_m <- ll; best_lam_m <- lam
        }
        
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta0 = beta0, Bfull = Bfull, beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi,
                           ll = ll, bic = bic, df = df, em_iter = em_it)
          if (grid_print_best) {
            cat(sprintf("\n[%s][GRID-BEST] %s | %d/%d | m=%d lam=%s | em_iter=%s ll=%s df=%d bic=%s | min(pi)=%s min(rho)=%s\n",
                        now_str(), method, grid_id, grid_total, m, fmt_num(lam, 6),
                        as.character(em_it), fmt_num(ll, 6), df, fmt_num(bic, 4),
                        fmt_num(minpi, 6), fmt_num(minrho, 6)))
          }
        }
      }
      
      if (grid_print_progress && (grid_id %% grid_trace_every == 0)) {
        t_el <- proc.time() - t_grid0
        cat(sprintf("[%s][GRID-PROG] %s | %d/%d | m=%d | bestBIC(m)=%s bestLam(m)=%s bestLL(m)=%s | bestBIC(all)=%s | elapsed=%ss\n",
                    now_str(), method, grid_id, grid_total, m,
                    fmt_num(best_bic_m, 4), fmt_num(best_lam_m, 6), fmt_num(best_ll_m, 6),
                    fmt_num(best_bic, 4), fmt_num(t_el["elapsed"], 1)))
      }
    }
  }
  
  if (!is.null(pb)) close(pb)
  best_all
}

# ============================================================
# Matching / Metrics (label switching 처리)
# ============================================================
# perms: permutation 생성(작은 m에서만 사용)
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

# match_to_true:
# - 추정 component와 true component를 SSE 최소가 되도록 permutation/부분집합 매칭
# - m_hat != m_true도 처리
match_to_true <- function(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true) {
  p <- nrow(B_true); m_true <- ncol(B_true); m_hat <- ncol(B_hat)
  best_sse <- Inf; best <- NULL
  
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
        Bpad <- matrix(0, p, m_true); pipad <- rep(0, m_true); s2pad <- rep(0, m_true)
        Bpad[, ts] <- Bh; pipad[ts] <- pi_hat[pp]; s2pad[ts] <- s2_hat[pp]
        best_sse <- sse
        best <- list(B = Bpad, pi = pipad, s2 = s2pad, m_hat = m_hat)
      }
    }
  }
  best
}

# compute_metrics:
# - b_hat = beta_tilde / rho 로 원모형 계수로 환산
# - MSE(b), MSE(pi), MSE(sigma^2)
# - 변수선택 지표:
#   * FPR/TPR: relevant set(1:10)
#   * FHR: common set(1:7)이 hetero로 잘못 선택된 비율
compute_metrics <- function(fit, truth, tol_sel = 1e-4) {
  if (is.null(fit)) return(list(mse_b = NA, FPR = NA, TPR = NA, FHR = NA, m_hat = NA,
                                mse_pi = NA, mse_sigma2 = NA))
  
  beta_tilde <- fit$beta_tilde; rho <- fit$rho; pi_hat <- fit$pi
  B_true <- truth$b_true; pi_true <- truth$pi_true; s2_true <- truth$sigma2_true
  
  s2_hat <- (1 / (rho + 1e-12))^2
  B_hat <- sweep(beta_tilde, 2, rho, "/")
  
  matched <- match_to_true(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true)
  Bh <- matched$B
  
  mse_b  <- mean((Bh - B_true)^2)
  mse_pi <- mean((matched$pi - pi_true)^2)
  mse_s2 <- mean((matched$s2 - s2_true)^2)
  
  p <- nrow(B_true)
  S_R <- truth$S_relevant; S_C <- truth$S_common
  
  sel_R <- which(apply(abs(B_hat), 1, function(v) any(v > tol_sel)))
  
  # hetero selection:
  # - HP면 Bfull 기준(직접 hetero 파라미터)
  # - 비-HP면 component별 편차(beta_tilde - rowmean) 기준
  if (!is.null(fit$method) && fit$method %in% c("Mix-HP-L", "Mix-HP-AL")) {
    Bfull <- fit$Bfull
    if (is.null(Bfull)) sel_H <- integer(0)
    else sel_H <- which(apply(abs(Bfull), 1, function(v) any(v > tol_sel)))
  } else {
    rowm <- rowMeans(beta_tilde)
    dev  <- sweep(beta_tilde, 1, rowm, "-")
    sel_H <- which(apply(abs(dev), 1, function(v) any(v > tol_sel)))
  }
  
  FPR <- length(intersect(sel_R, setdiff(1:p, S_R))) / (p - length(S_R))
  TPR <- length(intersect(sel_R, S_R)) / length(S_R)
  FHR <- length(intersect(sel_H, S_C)) / length(S_C)
  
  list(mse_b = mse_b, mse_pi = mse_pi, mse_sigma2 = mse_s2,
       FPR = FPR, TPR = TPR, FHR = FHR, m_hat = matched$m_hat)
}

# ============================================================
# Simulation driver (전체 루프 + TRACE)
# ============================================================
# run_sim_corrected:
# - rep마다 데이터 생성
# - method마다 fit_by_bic로 (m,lambda) 선택 및 적합
# - compute_metrics로 성능 기록
# - TRACE는 "어디서 오래 걸리는지" "수렴/붕괴가 있는지"를 미팅에서 바로 확인하기 위함
run_sim_corrected <- function(R = 20, n = 200, p = 60, snr = 50,
                              nlambda = 50,
                              max_em = 200, tol_em = 1e-5,
                              n_start = 10,
                              max_cd = 1000, tol_cd = 1e-7, min_cd = 10,
                              show_grid_bar = TRUE,
                              grid_trace_every = NULL,
                              grid_print_best = TRUE,
                              grid_print_progress = TRUE,
                              em_verbose = FALSE,
                              em_trace_every = 10) {
  
  res_all <- list()
  idx <- 1
  methods <- c("Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL")
  
  cat(sprintf("[%s] Running Simulation: n=%d, p=%d, SNR=%d, Reps=%d\n", now_str(), n, p, snr, R))
  t0 <- proc.time()
  
  for (r in 1:R) {
    cat(sprintf("\n[%s] ====================\n[%s] REP %d/%d START\n[%s] ====================\n",
                now_str(), now_str(), r, R, now_str()))
    
    dat <- simulate_dataset(n = n, p = p, snr = snr, seed = 2026 + r)
    
    for (meth in methods) {
      cat(sprintf("\n[%s] [REP %d] method=%s | BIC grid search START\n", now_str(), r, meth))
      t_m0 <- proc.time()
      
      fit <- fit_by_bic(dat$y, dat$X, meth,
                        nlambda = nlambda, lam_ratio = 1e-4,
                        n_start = n_start,
                        pilot_em = CFG$pilot_em,
                        max_em = max_em, tol_em = tol_em,
                        max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
                        verbose = FALSE, show_grid_bar = show_grid_bar,
                        grid_trace_every = grid_trace_every,
                        grid_print_best = grid_print_best,
                        grid_print_progress = grid_print_progress,
                        em_verbose = em_verbose,
                        em_trace_every = em_trace_every)
      
      if (is.null(fit)) {
        cat(sprintf("[%s] [REP %d] method=%s FAILED (no fit)\n", now_str(), r, meth))
        next
      }
      
      # method tag(hetero selection 분기용)
      fit$method <- meth
      
      met <- compute_metrics(fit, dat)
      
      t_m1 <- proc.time() - t_m0
      cat(sprintf("[%s] [REP %d] method=%s DONE | chosen(m)=%d lam=%s | em_iter=%s ll=%s bic=%s df=%d | elapsed=%ss\n",
                  now_str(), r, meth,
                  fit$m, fmt_num(fit$lam, 6),
                  as.character(fit$em_iter),
                  fmt_num(fit$ll, 6),
                  fmt_num(fit$bic, 4),
                  fit$df,
                  fmt_num(t_m1["elapsed"], 1)))
      
      res_all[[idx]] <- data.frame(
        rep = r, method = meth, snr = snr,
        chosen_m = fit$m, chosen_lam = fit$lam,
        ll = fit$ll, bic = fit$bic, df = fit$df, em_iter = fit$em_iter,
        mse_b = met$mse_b, mse_s2 = met$mse_sigma2, mse_pi = met$mse_pi,
        FPR = met$FPR, TPR = met$TPR, FHR = met$FHR, m_hat = met$m_hat
      )
      idx <- idx + 1
    }
    
    t_rep <- proc.time() - t0
    cat(sprintf("\n[%s] REP %d/%d END | total_elapsed=%ss\n", now_str(), r, R, fmt_num(t_rep["elapsed"], 1)))
  }
  
  out <- do.call(rbind, res_all)
  t1 <- proc.time() - t0
  cat(sprintf("\n[%s] ALL DONE | elapsed=%ss\n", now_str(), fmt_num(t1["elapsed"], 1)))
  out
}

# ============================================================
# Execution (LIGHT MODE)
# ============================================================
set.seed(999)

res <- run_sim_corrected(
  R       = CFG$R,
  n       = CFG$n,
  p       = CFG$p,
  snr     = CFG$snr,
  nlambda = CFG$nlambda,
  max_em  = CFG$max_em,
  tol_em  = CFG$tol_em,
  n_start = CFG$n_start,
  max_cd  = CFG$max_cd,
  tol_cd  = CFG$tol_cd,
  min_cd  = CFG$min_cd,
  show_grid_bar = TRUE,
  grid_trace_every = NULL,
  grid_print_best = TRUE,
  grid_print_progress = TRUE,
  em_verbose = FALSE,
  em_trace_every = 10
)

# ============================================================
# Summary table (method별 평균 성능 요약)
# ============================================================
library(dplyr)
summary_table <- res %>%
  group_by(method) %>%
  summarise(
    MSE_b  = mean(mse_b,  na.rm = TRUE),
    MSE_s2 = mean(mse_s2, na.rm = TRUE),
    FPR = mean(FPR, na.rm = TRUE) * 100,
    TPR = mean(TPR, na.rm = TRUE) * 100,
    FHR = mean(FHR, na.rm = TRUE) * 100,
    m_mode = as.integer(names(sort(table(m_hat), decreasing = TRUE)[1])),
    ll_mean = mean(ll, na.rm = TRUE),
    bic_mean = mean(bic, na.rm = TRUE),
    em_iter_mean = mean(em_iter, na.rm = TRUE)
  )
print(summary_table)

# ============================================================
# Plot helper (ggplot2 있으면 ggplot, 없으면 base R)
# ============================================================
plot_sim_results <- function(res, save_dir = NULL, prefix = "sim") {
  if (!is.data.frame(res) || nrow(res) == 0) {
    cat("plot_sim_results: empty res\n")
    return(invisible(NULL))
  }
  
  # base plot fallback
  plot_base <- function() {
    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)
    
    par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
    boxplot(mse_b ~ method, data = res, las = 2, main = "MSE(b) by method", ylab = "MSE(b)")
    boxplot(FPR ~ method, data = res, las = 2, main = "FPR by method", ylab = "FPR")
    boxplot(TPR ~ method, data = res, las = 2, main = "TPR by method", ylab = "TPR")
    boxplot(FHR ~ method, data = res, las = 2, main = "FHR by method", ylab = "FHR")
  }
  
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("tidyr", quietly = TRUE) ||
      !requireNamespace("dplyr", quietly = TRUE)) {
    cat("ggplot2/tidyr/dplyr not available. Using base plots.\n")
    plot_base()
    return(invisible(NULL))
  }
  
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  # metrics boxplots (long)
  res_long <- res %>%
    select(rep, method, mse_b, mse_s2, mse_pi, FPR, TPR, FHR, chosen_m, bic, ll, em_iter) %>%
    pivot_longer(cols = c(mse_b, mse_s2, mse_pi, FPR, TPR, FHR),
                 names_to = "metric", values_to = "value")
  
  p1 <- ggplot(res_long, aes(x = method, y = value)) +
    geom_boxplot() +
    facet_wrap(~ metric, scales = "free_y", ncol = 3) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    labs(title = "Simulation metrics by method", x = NULL, y = NULL)
  
  # chosen m distribution
  p2 <- ggplot(res, aes(x = factor(chosen_m))) +
    geom_bar() +
    facet_wrap(~ method, ncol = 2) +
    theme_bw() +
    labs(title = "Chosen m distribution (by method)", x = "chosen m", y = "count")
  
  # BIC / LL
  p3 <- ggplot(res, aes(x = method, y = bic)) +
    geom_boxplot() +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    labs(title = "BIC by method", x = NULL, y = "BIC")
  
  p4 <- ggplot(res, aes(x = method, y = ll)) +
    geom_boxplot() +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    labs(title = "Log-likelihood by method", x = NULL, y = "LL")
  
  print(p1); print(p2); print(p3); print(p4)
  
  if (!is.null(save_dir)) {
    if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    ggsave(file.path(save_dir, paste0(prefix, "_metrics_box.png")), p1, width = 12, height = 7)
    ggsave(file.path(save_dir, paste0(prefix, "_chosen_m_bar.png")), p2, width = 10, height = 6)
    ggsave(file.path(save_dir, paste0(prefix, "_bic_box.png")), p3, width = 8, height = 5)
    ggsave(file.path(save_dir, paste0(prefix, "_ll_box.png")), p4, width = 8, height = 5)
  }
  
  invisible(list(p_metrics = p1, p_m = p2, p_bic = p3, p_ll = p4))
}

# 시각화
plot_sim_results(res)
