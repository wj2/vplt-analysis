
data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  
  // prior data
  real prior_eps_mean;
  real<lower=0> prior_eps_var;
  real<lower=0> prior_salience_var_mean;
  real<lower=0> prior_salience_var_var;
  real<lower=0> prior_bias_mean_mean;
  real<lower=0> prior_bias_mean_var;
  real<lower=0> prior_bias_var_mean;
  real<lower=0> prior_bias_var_var;

  // main data
  int imgs[N, L, K] ; // image selection matrix
  int img_cats[L]; // indicator for whether image is novel or familiar
  matrix[N, K] views; // number of previous views
  int y[N]; // outcome vector
}

parameters {
  // prior-related
  real<lower=0> salience_var;
  real<lower=0> bias_mean;
  real<lower=0> bias_var;

  // data-related
  vector[L] sal_raw; // image inherent saliences
  vector[K - 1] bias_raw; // target bias terms
  real eps; // novelty bias
}

transformed parameters {
  vector<lower=0>[K - 1] bias;
  vector[L] s;
  bias = bias_mean + bias_var*bias_raw;
  s = eps*to_vector(img_cats) + salience_var*sal_raw;
}

model {
  // var declarations
  matrix[N, K] img_s;
  matrix[N, K] outcome_evidence;
  vector[K] simplex_evidence;

  // priors
  eps ~ normal(prior_eps_mean, prior_eps_var);
  salience_var ~ normal(prior_salience_var_mean, prior_salience_var_var);
  bias_var ~ normal(prior_bias_var_mean, prior_bias_var_var);
  bias_mean ~ normal(prior_bias_mean_mean, prior_bias_mean_var);
  
  bias_raw ~ normal(0, 1);
  sal_raw ~ normal(0, 1);
  
  // setup
  for (k in 1:K) {
    outcome_evidence[:, k] = to_matrix(imgs[:, :, k])*s;
  }
  
  for(k in 1:(K - 1)) {
    outcome_evidence[:, k] = outcome_evidence[:, k] + bias[k];
  }
  
  for (n in 1:N) {
    outcome_evidence[n, K] = 1;
    simplex_evidence = outcome_evidence[n]'/sum(outcome_evidence[n]);
    y[n] ~ categorical(simplex_evidence);
  }
}
