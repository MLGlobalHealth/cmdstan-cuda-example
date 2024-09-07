functions {
  #include headers.stan
}
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
transformed data {

}
parameters {
  real<lower=0> r;
  real<lower=0> a;
  real<lower=0> b;
}
model {
  r ~ normal(0, 1);
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  target += cu_beta_neg_binomial_lpmf(y | r, a, b);
}
generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = beta_neg_binomial_lpmf(y[i] | r, a, b);
  }
}

