data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  int<lower=0> D; // number of days of data
  
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
  matrix[N, K] views; // number of previous views
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
  vector[D] eps_raw; // novelty bias
}

transformed parameters {
  vector[D] bias1;
  vector[D] bias2;
  vector[D] eps;
  matrix[D, L] s;
  matrix[D, L] d_cats;
  eps = eps_mean + eps_var*eps_raw;
  bias1 = bias1_mean + bias1_var*bias1_raw;
  bias2 = bias2_mean + bias2_var*bias2_raw;
  for (d in 1:D) {
    s[d] = (img_cats*eps[d] + salience_var*sal_raw)';
  }
}

model {
  // var declarations
  real outcome_evidence[N, D, K];
  int d;
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
  eps_raw ~ normal(0, 1);
  
  // setup
  for (k in 1:K) {
    outcome_evidence[:, :, k] = to_array_2d(to_matrix(imgs[:, :, k])*s');
  }
  
  for (n in 1:N) {
    d = day[n];
    oe = to_vector(outcome_evidence[n, d]);
    oe[1] += bias1[d];
    oe[2] += bias2[d];
    y[n] ~ categorical_logit(oe);
  }
}
