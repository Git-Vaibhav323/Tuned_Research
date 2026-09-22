# =============================================================================
# ResearchPilot — Phase 2 DA2 (ENHANCED v3)
# SCRIPT 00: Enriched Feature Engineering
# =============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(stringr); library(tidyr)
})
set.seed(42)
cat("=== 00: Enriched Feature Engineering ===\n\n")

# ── 0. Root ────────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()
cat(sprintf("Root: %s\n\n", root))

data_in <- file.path(root, "data", "final", "final_dataset.csv")
eng_in  <- file.path(root, "data", "ml_r",  "engineered_features.csv")
out_dir <- file.path(root, "data", "ml_r")

# ── 1. Load ────────────────────────────────────────────────────────────────────
cat("Loading data...\n")
df     <- read_csv(data_in, show_col_types = FALSE)
df_eng <- read_csv(eng_in,  show_col_types = FALSE)
cat(sprintf("  %d papers loaded\n\n", nrow(df)))

# Build text vectors as plain character vectors (avoid any dplyr conversion)
title_low    <- tolower(as.character(df$title))
abstract_low <- tolower(as.character(df$abstract))
text_full    <- paste(title_low, abstract_low, sep = " ")
kw_low       <- tolower(as.character(df$keywords_clean))
cn_low       <- tolower(as.character(df$concepts_clean))
doi_clean    <- tolower(ifelse(is.na(df$doi), "", as.character(df$doi)))

# Helper: detect pattern → integer 0/1 vector
gd <- function(pattern, x) as.integer(grepl(pattern, x, perl = TRUE, ignore.case = TRUE))

# ── 2. Domain binary features (one at a time to avoid data.frame hang) ────────
cat("Part A: Domain binary indicators...\n")

# Medical / Biology
dom_medical    <- gd("medical|clinical|patient|hospital|disease|cancer|covid|drug therapy|diagnosis|radiolog", text_full); cat("  dom_medical:", sum(dom_medical), "\n")
dom_biology    <- gd("protein|genome|\\bdna\\b|\\brna\\b|molecular biology|bioinform|metabolom|genome sequenc", text_full); cat("  dom_biology:", sum(dom_biology), "\n")
dom_health     <- gd("public health|mental health|epidem|vaccine|pandemic|wellbeing", text_full); cat("  dom_health:", sum(dom_health), "\n")
dom_cv_medical <- gd("medical image|\\bmri\\b|ct scan|x-ray|x ray|ultrasound|histolog|fundus|retinal", text_full); cat("  dom_cv_medical:", sum(dom_cv_medical), "\n")
# Vision
dom_vision     <- gd("object detect|image segment|convolutional|\\bcnn\\b|\\byolo\\b|3d gaussian|novel.view|rendering|computer vision", text_full); cat("  dom_vision:", sum(dom_vision), "\n")
dom_vision2    <- gd("\\bimage\\b|visual feature|pixel|resnet|vgg|image recognit|image classif", text_full); cat("  dom_vision2:", sum(dom_vision2), "\n")
# NLP / LLM
dom_nlp        <- gd("natural language processing|text classif|sentiment analysis|named entity|machine translation|text summariz", text_full); cat("  dom_nlp:", sum(dom_nlp), "\n")
dom_llm        <- gd("large language model|chatgpt|\\bgpt\\b|\\bllm\\b|\\bbert\\b|\\btransformer\\b|prompt tuning|instruction tuning|\\bllama\\b", text_full); cat("  dom_llm:", sum(dom_llm), "\n")
dom_generative <- gd("generative model|diffusion model|stable diffusion|image generation|text.to.image|dall.e", text_full); cat("  dom_generative:", sum(dom_generative), "\n")
# Education
dom_education  <- gd("\\beducation\\b|\\bstudent\\b|\\bteacher\\b|classroom|curriculum|pedagog|higher education|e-learning|\\bmooc\\b", text_full); cat("  dom_education:", sum(dom_education), "\n")
# Robotics / IoT
dom_robotics   <- gd("\\brobot\\b|autonomous vehicle|\\bdrone\\b|internet of things|embedded system|real-time control|\\bslam\\b", text_full); cat("  dom_robotics:", sum(dom_robotics), "\n")
# Math / Theory
dom_math       <- gd("\\btheorem\\b|\\bproof\\b|convergence|bayesian inference|variational inference|stochastic process", text_full); cat("  dom_math:", sum(dom_math), "\n")
dom_quantum    <- gd("\\bquantum\\b|\\bqubit\\b|quantum circuit|quantum computing|\\bnisq\\b", text_full); cat("  dom_quantum:", sum(dom_quantum), "\n")
# Survey
dom_survey     <- gd("^survey|^review|^overview|^tutorial|^comprehensive survey|^systematic review", title_low); cat("  dom_survey:", sum(dom_survey), "\n")
dom_review_any <- gd("systematic review|literature review|survey of|overview of|comprehensive review", text_full); cat("  dom_review_any:", sum(dom_review_any), "\n")
# Graph / RL / Speech
dom_graph      <- gd("graph neural network|knowledge graph|graph convolut|network embedding|node classif", text_full); cat("  dom_graph:", sum(dom_graph), "\n")
dom_rl         <- gd("reinforcement learning|reward function|policy gradient|q-learning|markov decision|actor.critic|\\bppo\\b|\\bdqn\\b", text_full); cat("  dom_rl:", sum(dom_rl), "\n")
dom_speech     <- gd("speech recognit|speaker identif|acoustic model|\\bwav2vec\\b|spoken language|speech synthesis", text_full); cat("  dom_speech:", sum(dom_speech), "\n")
# Security / XAI
dom_security   <- gd("adversarial attack|adversarial example|federated learning|differential privacy|\\bwatermark\\b|model inversion", text_full); cat("  dom_security:", sum(dom_security), "\n")
dom_xai        <- gd("explainab|interpretab|\\bxai\\b|\\bshap\\b|\\blime\\b|feature attribution|\\bfairness\\b|algorithmic bias", text_full); cat("  dom_xai:", sum(dom_xai), "\n")
# Science / Dataset
dom_science    <- gd("protein structure|alphafold|molecular dynamics|quantum chemistry|materials science|physics-informed", text_full); cat("  dom_science:", sum(dom_science), "\n")
dom_dataset    <- gd("\\bdataset\\b|\\bbenchmark\\b|\\bcorpus\\b|annotation pipeline|data collection|crowdsourc", text_full); cat("  dom_dataset:", sum(dom_dataset), "\n")
# Concept-level signals
cn_medicine    <- gd("medicine|medical|biology|health care", cn_low); cat("  cn_medicine:", sum(cn_medicine), "\n")
cn_ai_core     <- gd("artificial intelligence|machine learning|deep learning", cn_low); cat("  cn_ai_core:", sum(cn_ai_core), "\n")
cn_vision      <- gd("computer vision|image|rendering", cn_low); cat("  cn_vision:", sum(cn_vision), "\n")
cat("  Part A done.\n\n")

# ── 3. Publisher / DOI signals ────────────────────────────────────────────────
cat("Part B: Publisher signals...\n")
pub_plos      <- gd("10\\.1371", doi_clean); cat("  pub_plos:", sum(pub_plos), "\n")
pub_mdpi      <- gd("10\\.3390", doi_clean); cat("  pub_mdpi:", sum(pub_mdpi), "\n")
pub_frontiers <- gd("10\\.3389", doi_clean); cat("  pub_frontiers:", sum(pub_frontiers), "\n")
pub_bmc       <- gd("10\\.1186", doi_clean); cat("  pub_bmc:", sum(pub_bmc), "\n")
pub_hindawi   <- gd("10\\.1155", doi_clean); cat("  pub_hindawi:", sum(pub_hindawi), "\n")
pub_ieee      <- gd("10\\.1109", doi_clean); cat("  pub_ieee:", sum(pub_ieee), "\n")
pub_acm       <- gd("10\\.1145", doi_clean); cat("  pub_acm:", sum(pub_acm), "\n")
pub_elsevier  <- gd("10\\.1016", doi_clean); cat("  pub_elsevier:", sum(pub_elsevier), "\n")
pub_springer  <- gd("10\\.1007", doi_clean); cat("  pub_springer:", sum(pub_springer), "\n")
pub_wiley     <- gd("10\\.1002", doi_clean); cat("  pub_wiley:", sum(pub_wiley), "\n")
pub_nature    <- gd("10\\.1038", doi_clean); cat("  pub_nature:", sum(pub_nature), "\n")
pub_science   <- gd("10\\.1126", doi_clean); cat("  pub_science:", sum(pub_science), "\n")
pub_oup       <- gd("10\\.1093", doi_clean); cat("  pub_oup:", sum(pub_oup), "\n")
pub_arxiv     <- gd("10\\.48550|arxiv", doi_clean); cat("  pub_arxiv:", sum(pub_arxiv), "\n")
pub_zenodo    <- gd("10\\.5281",  doi_clean); cat("  pub_zenodo:", sum(pub_zenodo), "\n")
pub_dagstuhl  <- gd("10\\.4230",  doi_clean); cat("  pub_dagstuhl:", sum(pub_dagstuhl), "\n")
pub_acl       <- gd("10\\.18653", doi_clean); cat("  pub_acl:", sum(pub_acl), "\n")
pub_no_doi    <- as.integer(is.na(df$doi));   cat("  pub_no_doi:", sum(pub_no_doi), "\n")
cat("  Part B done.\n\n")

# ── 4. TF-IDF features ───────────────────────────────────────────────────────
cat("Part C: TF-IDF features...\n")
X_tfidf   <- NULL
n_tfidf   <- 0L
top_terms <- character(0)

if (requireNamespace("tidytext", quietly = TRUE)) {
  suppressPackageStartupMessages(library(tidytext))

  doc_df <- data.frame(doc_id = seq_len(nrow(df)),
                       text   = text_full,
                       stringsAsFactors = FALSE)

  tokens <- doc_df %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) >= 4, !grepl("^[0-9]", word)) %>%
    count(doc_id, word, name = "n")

  tfidf_df <- tokens %>%
    bind_tf_idf(word, doc_id, n)

  n_docs    <- nrow(df)
  top_terms <- tfidf_df %>%
    group_by(word) %>%
    summarise(total    = sum(tf_idf),
              doc_freq = n_distinct(doc_id), .groups = "drop") %>%
    filter(doc_freq >= 5, doc_freq <= n_docs * 0.85) %>%
    arrange(desc(total)) %>%
    head(150) %>%
    pull(word)

  cat(sprintf("  Vocabulary: %d terms selected\n", length(top_terms)))

  tfidf_wide <- tfidf_df %>%
    filter(word %in% top_terms) %>%
    select(doc_id, word, tf_idf) %>%
    pivot_wider(names_from  = word,
                values_from = tf_idf,
                values_fill = 0,
                names_prefix = "tf_")

  n_tfidf   <- length(top_terms)
  col_names <- paste0("tf_", top_terms)
  X_tfidf   <- matrix(0.0, nrow = nrow(df), ncol = n_tfidf,
                      dimnames = list(NULL, col_names))
  ids <- tfidf_wide$doc_id
  ids <- ids[ids >= 1L & ids <= nrow(df)]
  X_tfidf[ids, ] <- as.matrix(
    tfidf_wide[match(ids, tfidf_wide$doc_id), col_names, drop = FALSE])
  cat(sprintf("  TF-IDF: %d × %d\n\n", nrow(X_tfidf), ncol(X_tfidf)))
} else {
  cat("  tidytext not available — skipped\n\n")
}

# ── 5. Structural features ───────────────────────────────────────────────────
cat("Part D: Structural features...\n")
keep_s <- intersect(
  c("title_length","abstract_length","keyword_count","concept_count","paper_age",
    "title_word_count","abstract_word_count","title_to_abstract_ratio",
    "text_richness","recency_score","abstract_keyword_overlap","recent_paper"),
  names(df_eng))
struct_df <- df_eng[, keep_s, drop = FALSE] %>%
  mutate(across(where(is.logical), as.integer))
cat(sprintf("  %d structural features\n\n", ncol(struct_df)))

# ── 6. Assemble — build column by column (avoids data.frame() hang) ───────────
cat("Assembling feature matrix...\n")
out <- struct_df

# Domain features
for (nm in c("dom_medical","dom_biology","dom_health","dom_cv_medical",
             "dom_vision","dom_vision2","dom_nlp","dom_llm","dom_generative",
             "dom_education","dom_robotics","dom_math","dom_quantum",
             "dom_survey","dom_review_any","dom_graph","dom_rl","dom_speech",
             "dom_security","dom_xai","dom_science","dom_dataset",
             "cn_medicine","cn_ai_core","cn_vision")) {
  out[[nm]] <- get(nm)
}

# Publisher features
for (nm in c("pub_plos","pub_mdpi","pub_frontiers","pub_bmc","pub_hindawi",
             "pub_ieee","pub_acm","pub_elsevier","pub_springer","pub_wiley",
             "pub_nature","pub_science","pub_oup","pub_arxiv","pub_zenodo",
             "pub_dagstuhl","pub_acl","pub_no_doi")) {
  out[[nm]] <- get(nm)
}

# TF-IDF
if (!is.null(X_tfidf)) {
  tfidf_frame <- as.data.frame(X_tfidf, stringsAsFactors = FALSE)
  out <- bind_cols(out, tfidf_frame)
}

# Target
out$oa_category <- as.character(df$oa_category)

total_f <- ncol(out) - 1L
cat(sprintf("Final: %d rows × %d features + target\n", nrow(out), total_f))
cat(sprintf("  Structural : %d\n", length(keep_s)))
cat(sprintf("  Domain     : 25\n"))
cat(sprintf("  Publisher  : 18\n"))
cat(sprintf("  TF-IDF     : %d\n", n_tfidf))

# ── 7. Quick discriminability check ──────────────────────────────────────────
cat("\nDiscriminability check (publisher means by OA class):\n")
check_cols <- c("pub_ieee","pub_plos","pub_mdpi","pub_bmc","pub_elsevier","pub_acm",
                "pub_frontiers","pub_hindawi","dom_medical","dom_vision","dom_llm","dom_education")
check_cols <- intersect(check_cols, names(out))
for (col in check_cols) {
  v <- tapply(out[[col]], out$oa_category, mean)
  cat(sprintf("  %-20s closed=%.3f  fully=%.3f  partial=%.3f\n",
              col, v["closed"], v["fully_open"], v["partially_open"]))
}

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path <- file.path(out_dir, "enriched_features_v2.csv")
write_csv(out, out_path)
cat(sprintf("\nSaved: %s  (%d × %d)\n", out_path, nrow(out), ncol(out)))

meta <- data.frame(
  feature = setdiff(names(out), "oa_category"),
  group   = case_when(
    startsWith(setdiff(names(out),"oa_category"), "tf_")  ~ "tfidf",
    startsWith(setdiff(names(out),"oa_category"), "dom_") |
      startsWith(setdiff(names(out),"oa_category"), "cn_") ~ "domain",
    startsWith(setdiff(names(out),"oa_category"), "pub_") ~ "publisher",
    TRUE ~ "structural"
  )
)
write_csv(meta, file.path(out_dir, "enriched_feature_metadata.csv"))
cat("\nFeature groups:\n"); print(table(meta$group))
cat("\n=== 00 COMPLETE ===\n")
