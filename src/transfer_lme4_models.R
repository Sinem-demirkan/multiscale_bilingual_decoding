#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
})

parse_args = function() {
  args = commandArgs(trailingOnly = TRUE)
  out = list()

  i = 1
  while (i <= length(args)) {
    key = args[[i]]
    if (!startsWith(key, "--")) {
      stop(paste("Unexpected argument:", key))
    }
    if (i == length(args)) {
      stop(paste("Missing value for", key))
    }
    out[[sub("^--", "", key)]] = args[[i + 1]]
    i = i + 2
  }

  required = c("transfer", "self-decoding", "qc", "out-dir")
  missing = setdiff(required, names(out))
  if (length(missing) > 0) {
    stop(paste("Missing required arguments:", paste(missing, collapse = ", ")))
  }

  out
}

variance_table = function(model) {
  vc = as.data.frame(VarCorr(model))
  rows = vc[vc$grp %in% c("teacher", "learner", "Residual"), c("grp", "vcov")]
  rows$component = ifelse(rows$grp == "teacher", "Teacher",
                   ifelse(rows$grp == "learner", "Learner", "Pairwise residual"))
  rows$variance = rows$vcov
  rows$pct = 100 * rows$variance / sum(rows$variance)
  rows[, c("component", "variance", "pct")]
}

random_effect_table = function(model) {
  teacher = ranef(model)$teacher
  learner = ranef(model)$learner

  teacher$subject = rownames(teacher)
  learner$subject = rownames(learner)

  names(teacher)[names(teacher) == "(Intercept)"] = "effect"
  names(learner)[names(learner) == "(Intercept)"] = "effect"

  teacher$role = "Teacher"
  learner$role = "Learner"

  out = rbind(
    teacher[, c("subject", "role", "effect")],
    learner[, c("subject", "role", "effect")]
  )
  rownames(out) = NULL
  out
}

fixed_effect_table = function(model) {
  coef_table = as.data.frame(summary(model)$coefficients)
  coef_table$term = rownames(coef_table)
  names(coef_table) = c("estimate", "std_error", "df", "t_value", "term")

  ci = suppressMessages(confint(model, method = "Wald"))
  ci = as.data.frame(ci)
  ci$term = rownames(ci)
  names(ci) = c("conf_low", "conf_high", "term")

  merge(coef_table, ci, by = "term", all.x = TRUE)
}

zscore = function(x) {
  as.numeric(scale(x))
}

args = parse_args()
out_dir = args[["out-dir"]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

transfer = read.csv(args[["transfer"]], stringsAsFactors = FALSE)
self = read.csv(args[["self-decoding"]], stringsAsFactors = FALSE)
qc = read.csv(args[["qc"]], stringsAsFactors = FALSE)

self = self[, c("subject", "balanced_accuracy")]
names(self)[names(self) == "balanced_accuracy"] = "selfacc"
qc = qc[, c("subject", "mean_tsnr")]

subject_covariates = merge(self, qc, by = "subject")
subject_covariates$selfacc_z = zscore(subject_covariates$selfacc)
subject_covariates$tsnr_z = zscore(subject_covariates$mean_tsnr)

df = transfer
df = df[df$train_subject != df$test_subject, ]
df$teacher = df$train_subject
df$learner = df$test_subject

df = merge(df, subject_covariates, by.x = "teacher", by.y = "subject", all.x = TRUE)
names(df)[names(df) == "selfacc"] = "teacher_selfacc"
names(df)[names(df) == "mean_tsnr"] = "teacher_tsnr"
names(df)[names(df) == "selfacc_z"] = "teacher_selfacc_z"
names(df)[names(df) == "tsnr_z"] = "teacher_tsnr_z"

df = merge(df, subject_covariates, by.x = "learner", by.y = "subject", all.x = TRUE)
names(df)[names(df) == "selfacc"] = "learner_selfacc"
names(df)[names(df) == "mean_tsnr"] = "learner_tsnr"
names(df)[names(df) == "selfacc_z"] = "learner_selfacc_z"
names(df)[names(df) == "tsnr_z"] = "learner_tsnr_z"

df = df[complete.cases(df[, c(
  "balanced_accuracy",
  "teacher_selfacc", "learner_selfacc",
  "teacher_tsnr", "learner_tsnr"
)]), ]

unadjusted = lmer(
  balanced_accuracy ~ 1 + (1 | teacher) + (1 | learner),
  data = df,
  REML = TRUE
)

adjusted = lmer(
  balanced_accuracy ~
    teacher_selfacc_z + learner_selfacc_z +
    teacher_tsnr_z + learner_tsnr_z +
    teacher_selfacc_z:teacher_tsnr_z +
    learner_selfacc_z:learner_tsnr_z +
    (1 | teacher) + (1 | learner),
  data = df,
  REML = TRUE
)

write.csv(df, file.path(out_dir, "lme4_transfer_design.csv"), row.names = FALSE)

write.csv(variance_table(unadjusted), file.path(out_dir, "lme4_unadjusted_variance_components.csv"), row.names = FALSE)
write.csv(random_effect_table(unadjusted), file.path(out_dir, "lme4_unadjusted_subject_effects_long.csv"), row.names = FALSE)

write.csv(variance_table(adjusted), file.path(out_dir, "lme4_variance_components.csv"), row.names = FALSE)
write.csv(random_effect_table(adjusted), file.path(out_dir, "lme4_subject_effects_long.csv"), row.names = FALSE)
write.csv(fixed_effect_table(adjusted), file.path(out_dir, "lme4_fixed_effects.csv"), row.names = FALSE)

capture.output(summary(unadjusted), file = file.path(out_dir, "lme4_unadjusted_summary.txt"))
capture.output(summary(adjusted), file = file.path(out_dir, "lme4_adjusted_summary.txt"))
