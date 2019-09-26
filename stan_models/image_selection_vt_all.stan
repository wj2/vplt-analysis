data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  int<lower=0> D; // number of days of data
  int<lower=0> V; // number of view bins
  
  // prior data
  real prior_eps_mean_mean;
  real<lower=0> prior_eps_var_mean;
  real prior_eps_mean_var;
  real<lower=0> prior_eps_var_var;
  
  real prior_salience_var_mean;
  real<lower=0> prior_salience_var_var;
  real prior_bias_mean_mean;
  real<lower=0> prior_bias_mean_var;
  real<lower=0> prior_bias_var_mean;
  real<lower=0> prior_bias_var_var;

  // main data
  int imgs[N, L, K] ; // image selection matrix
  vector[L] img_cats; // indicator for whether image is novel or familiar
  int views[N, K]; // number of previous views
  int y[N]; // outcome vector
  int<lower=1> day[N]; // day vector
}

parameters {
  // prior-related
  real<lower=0> salience_var;
  real bias1_mean;
  real bias2_mean;
  real<lower=0> bias1_var;
  real<lower=0> bias2_var;
  real eps_mean;
  real<lower=0> eps_var;

  // data-related
  vector[L] sal_raw; // image inherent saliences
  vector[D] bias1_raw; // target bias terms
  vector[D] bias2_raw; // target bias terms
  real eps_raw[D, V]; // novelty bias
}

transformed parameters {
  vector[D] bias1;
  vector[D] bias2;
  matrix[D, V] eps;
  vector[L] sal;
  eps = eps_mean + eps_var*to_matrix(eps_raw);
  bias1 = bias1_mean + bias1_var*bias1_raw;
  bias2 = bias2_mean + bias2_var*bias2_raw;
  sal = salience_var*sal_raw;
}

model {
  // var declarations
  real s[L, D, V];
  int d;
  int v1;
  int v2;
  vector[K] oe;

  // priors
  salience_var ~ normal(prior_salience_var_mean, prior_salience_var_var);
  bias1_var ~ normal(prior_bias_var_mean, prior_bias_var_var);
  bias1_mean ~ normal(prior_bias_mean_mean, prior_bias_mean_var);
  bias2_var ~ normal(prior_bias_var_mean, prior_bias_var_var);
  bias2_mean ~ normal(prior_bias_mean_mean, prior_bias_mean_var);
  eps_mean ~ normal(prior_eps_mean_mean, prior_eps_mean_var);
  eps_var ~ normal(prior_eps_var_mean, prior_eps_var_var);
  
  bias1_raw ~ normal(0, 1);
  bias2_raw ~ normal(0, 1);
  sal_raw ~ normal(0, 1);
  for (di in 1:D) {
    eps_raw[di] ~ normal(0, 1);
  }

  for (l in 1:L) {
    s[l, :, :] = to_array_2d(eps*img_cats[l] + sal[l]);
  }

  oe[3] = 0;
  
  // model  
  for (n in 1:N) {
    d = day[n];
    v1 = views[n, 1];
    v2 = views[n, 2];
    oe[1] = to_row_vector(imgs[n, :, 1])*to_vector(s[:, d, v1]) + bias1[d];
    oe[2] = to_row_vector(imgs[n, :, 2])*to_vector(s[:, d, v2]) + bias2[d];
    y[n] ~ categorical_logit(oe);
  }
}
