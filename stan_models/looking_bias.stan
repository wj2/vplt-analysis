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
  real prior_look_var_mean;
  real<lower=0> prior_look_var_var; 

  // main data
  int imgs[N, L, K] ; // image selection matrix
  vector[L] img_cats; // indicator for whether image is novel or familiar
  int views[N, K]; // number of previous views
  real y[N]; // outcome vector
  int<lower=1> day[N]; // day vector
}

parameters {
  // prior-related
  real<lower=0> salience_var;
  real bias_mean;
  real<lower=0> bias_var;
  real eps_mean;
  real<lower=0> eps_var;
  real<lower=0> look_var_var;

  // data-related
  vector[L] sal_raw; // image inherent saliences
  vector[D] bias_raw; // target bias terms
  vector<lower=0>[D] look_var_raw;
  real eps_raw[D, V]; // novelty bias
}

transformed parameters {
  vector[D] bias;
  matrix[D, V] eps;
  vector<lower=0>[L] sal;
  vector<lower=0>[D] look_var;
  eps = eps_mean + eps_var*to_matrix(eps_raw);
  bias = bias_mean + bias_var*bias_raw;
  sal = salience_var*sal_raw;
  look_var = look_var_var*look_var_raw;
}

model {
  // var declarations
  real s[L, D, V];
  int d;
  int v1;
  int v2;
  vector[2] oe;

  // priors
  salience_var ~ normal(prior_salience_var_mean, prior_salience_var_var);
  bias_var ~ normal(prior_bias_var_mean, prior_bias_var_var);
  bias_mean ~ normal(prior_bias_mean_mean, prior_bias_mean_var);
  eps_mean ~ normal(prior_eps_mean_mean, prior_eps_mean_var);
  eps_var ~ normal(prior_eps_var_mean, prior_eps_var_var);
  look_var_var ~ normal(prior_look_var_mean, prior_look_var_var);
  
  bias_raw ~ normal(0, 1);
  sal_raw ~ normal(0, 1);
  look_var_raw ~ normal(0, 1);
  for (di in 1:D) {
    eps_raw[di] ~ normal(0, 1);
  }

  for (l in 1:L) {
    s[l, :, :] = to_array_2d(eps*img_cats[l] + sal[l]);
  }

  // model  
  for (n in 1:N) {
    d = day[n];
    v1 = views[n, 1];
    v2 = views[n, 2];
    oe[1] = to_row_vector(imgs[n, :, 1])*to_vector(s[:, d, v1]) + bias[d];
    oe[2] = to_row_vector(imgs[n, :, 2])*to_vector(s[:, d, v2]);
    y[n] ~ normal(oe[1] - oe[2], look_var[d]);
  }
}
