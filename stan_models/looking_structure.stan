data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  int<lower=0> D; // number of days of data
  int<lower=0> V; // number of view bins
  int<lower=0> S; // number of saccades to model to
  
  // prior data
  real prior_sw_mean_mean;
  real<lower=0> prior_sw_mean_var;
  real prior_sw_var_mean;
  real<lower=0> prior_sw_var_var;

  real prior_nw_mean_mean;
  real<lower=0> prior_nw_mean_var;
  real prior_nw_var_mean;
  real<lower=0> prior_nw_var_var;

  real prior_bw_mean_mean;
  real<lower=0> prior_bw_mean_var;
  real prior_bw_var_mean;
  real<lower=0> prior_bw_var_var;
  
  real prior_eps_var_mean;
  real<lower=0> prior_eps_var_var;

  real prior_salience_var_mean;
  real<lower=0> prior_salience_var_var;
  
  real<lower=0> prior_bias_var_mean;
  real<lower=0> prior_bias_var_var;
  
  real prior_look_var_mean;
  real<lower=0> prior_look_var_var; 

  // main data
  int imgs[N, L, K] ; // image selection matrix
  vector[L] img_cats; // indicator for whether image is novel or familiar
  matrix[N, K - 1] trial_nov;
  int views[N, K]; // number of previous views
  int y[N, S]; // outcome vector
  int<lower=1> day[N]; // day vector
}

parameters {
  // prior-related
  real<lower=0> salience_var;
  real<lower=0> eps_var;

  vector[S] sw_mean;
  vector[S] sw_var;
  
  vector[S] nw_mean;
  vector[S] nw_var;
  
  vector[S] bw_mean;
  vector[S] bw_var;

  // data-related
  vector[L] sal_raw; // image inherent saliences
  real eps_raw[D, V]; // novelty bias

  // trajectory-related
  matrix[S, D] sal_weight_raw;
  real bias_weight_raw[S, D, K - 1]; 
  matrix[S, D] nov_weight_raw;
}

transformed parameters {
  matrix[D, V] eps;
  vector[L] sal;

  matrix[S, D] sal_weight;
  real bias_weight[S, D, K - 1]; 
  matrix[S, D] nov_weight;
  
  eps = 1 + eps_var*to_matrix(eps_raw);
  sal = 1 + salience_var*sal_raw;

  for (d in 1:D) {
    sal_weight[:, d] = sw_mean + sw_var .* sal_weight_raw[:, d];
    nov_weight[:, d] = nw_mean + nw_var .* nov_weight_raw[:, d];
    for (k in 1:K-1) {
      bias_weight[:, d, k] = to_array_1d(bw_mean
					 + bw_var
					 .* to_vector(bias_weight_raw[:, d, k]));
    }
  }
}

model {
  // var declarations
  real nov_contrib[L, D, V];
  int d;
  int v_i;
  matrix[S, K] oe;
  real img_sal;
  real img_nov;

  // priors
  salience_var ~ normal(prior_salience_var_mean, prior_salience_var_var);
  eps_var ~ normal(prior_eps_var_mean, prior_eps_var_var);

  sw_mean ~ normal(prior_sw_mean_mean, prior_sw_mean_var);
  sw_var ~ normal(prior_sw_var_mean, prior_sw_var_var);

  nw_mean ~ normal(prior_nw_mean_mean, prior_nw_mean_var);
  nw_var ~ normal(prior_nw_var_mean, prior_nw_var_var);

  bw_mean ~ normal(prior_bw_mean_mean, prior_bw_mean_var);
  bw_var ~ normal(prior_bw_var_mean, prior_bw_var_var);
  
  sal_raw ~ normal(0, 1);
  for (v in 1:V) {
    eps_raw[v] ~ normal(0, 1);
  }
  for (si in 1:S) {
    sal_weight_raw[si] ~ normal(0, 1);
    nov_weight_raw[si] ~ normal(0, 1);
    for (k in 1:K - 1) {
      bias_weight_raw[si, :, k] ~ normal(0, 1);
    }
  }

  // model  
  for (n in 1:N) {
    d = day[n];
    for (k in 1:K - 1) {
      v_i = views[n, k];
      img_sal = to_row_vector(imgs[n, :, k])*sal;
      img_nov = trial_nov[n, k]*eps[d, v_i];
      for (s in 1:S) {
	oe[s, k] = nov_weight[s, d]*img_nov;
	oe[s, k] += sal_weight[s, d]*img_sal;
	oe[s, k] += bias_weight[s, d, k];
      }
    }
    oe[:, K] = rep_vector(0, S);
    for (s in 1:S) {
      y[n, s] ~ categorical_logit(oe[s]');
    }
  }
}
