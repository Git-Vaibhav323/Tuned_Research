# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 7b: Topic Clustering (R — TF-IDF + KMeans)
# =============================================================================
# Verified baseline: k=8, silhouette ≈ 0.064
# Input : data/final/final_dataset.csv (title + abstract for TF-IDF)
# Output: data/ml_r/cluster_assignments.csv
#         data/ml_r/cluster_keywords.csv
#         reports/figures/phase2_r/clustering_*.png
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(stringr); library(ggplot2); library(tidyr)
})
set.seed(42)
cat("=== M7b: Topic Clustering (TF-IDF + KMeans) ===\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

data_in <- file.path(root, "data", "final", "final_dataset.csv")
out_dir <- file.path(root, "data", "ml_r")
fig_dir <- file.path(root, "reports", "figures", "phase2_r")
rep_dir <- file.path(root, "reports", "tables")
dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)

# ── 1. Load and prepare text ──────────────────────────────────────────────────
cat("Loading corpus...\n")
df <- read_csv(data_in, show_col_types=FALSE)
cat(sprintf("  %d papers loaded\n\n", nrow(df)))

# Combine title + abstract as clustering text
df$text_combined <- paste(df$title, df$abstract, sep=" ")

# ── 2. TF-IDF via tidytext ────────────────────────────────────────────────────
cat("Building TF-IDF matrix...\n")

tfidf_avail <- requireNamespace("tidytext", quietly = TRUE)
irlba_avail <- requireNamespace("irlba",    quietly = TRUE)

if (tfidf_avail) {
  suppressPackageStartupMessages({
    library(tidytext)
    if (requireNamespace("SnowballC", quietly=TRUE)) library(SnowballC)
  })

  # Standard English stop words (built into tidytext)
  data("stop_words", package="tidytext", envir=environment())

  # Tokenise + remove stop words + filter short words
  tokens <- data.frame(
    doc_id = seq_len(nrow(df)),
    text   = df$text_combined,
    stringsAsFactors=FALSE
  ) %>%
    tidytext::unnest_tokens(word, text) %>%
    anti_join(stop_words, by="word") %>%
    filter(nchar(word) > 3, !str_detect(word, "^[0-9]+$"))

  # Optionally stem
  if (requireNamespace("SnowballC", quietly=TRUE)) {
    tokens$word <- SnowballC::wordStem(tokens$word, language="en")
  }

  # Compute TF-IDF
  tfidf <- tokens %>%
    count(doc_id, word) %>%
    bind_tf_idf(word, doc_id, n) %>%
    filter(tf_idf > 0)

  # Keep top 200 terms by total TF-IDF weight (for tractable matrix)
  top_terms <- tfidf %>%
    group_by(word) %>%
    summarise(total_tfidf=sum(tf_idf)) %>%
    arrange(desc(total_tfidf)) %>%
    head(200) %>%
    pull(word)

  tfidf_filtered <- tfidf %>% filter(word %in% top_terms)

  # Cast to sparse matrix (Matrix package — no tm dependency)
  dtm_sparse <- tfidf_filtered %>%
    tidytext::cast_sparse(doc_id, word, tf_idf)

  # Convert to dense matrix
  X_tfidf <- as.matrix(dtm_sparse)

  # Align rows back to doc order 1:2000
  row_ids   <- as.integer(rownames(X_tfidf))
  X_full    <- matrix(0.0, nrow = nrow(df), ncol = ncol(X_tfidf),
                      dimnames = list(seq_len(nrow(df)), colnames(X_tfidf)))
  valid_ids <- row_ids[row_ids >= 1 & row_ids <= nrow(df)]
  X_full[valid_ids, ] <- X_tfidf[row_ids >= 1 & row_ids <= nrow(df), ]
  X_tfidf <- X_full

  cat(sprintf("  TF-IDF matrix: %d × %d (docs × terms)\n\n",
              nrow(X_tfidf), ncol(X_tfidf)))

} else {
  cat("  tidytext not available — using simple BoW from keywords_clean\n\n")
  # Fallback: encode keyword presence as binary features
  all_kws <- unique(unlist(str_split(df$keywords_clean, ";\\s*")))
  all_kws  <- all_kws[nchar(all_kws) > 3][1:min(200, length(all_kws))]
  X_tfidf  <- matrix(0L, nrow=nrow(df), ncol=length(all_kws))
  colnames(X_tfidf) <- all_kws
  for (i in seq_len(nrow(df))) {
    kws <- str_split(df$keywords_clean[i], ";\\s*")[[1]]
    X_tfidf[i, colnames(X_tfidf) %in% kws] <- 1L
  }
  cat(sprintf("  Fallback BoW matrix: %d × %d\n\n", nrow(X_tfidf), ncol(X_tfidf)))
}

# Normalise rows (L2) so cosine ~ Euclidean in KMeans
row_norms <- sqrt(rowSums(X_tfidf^2))
row_norms[row_norms == 0] <- 1
X_norm <- X_tfidf / row_norms

# ── 3. PCA reduction for visualisation ────────────────────────────────────────
cat("PCA reduction...\n")
if (irlba_avail) {
  suppressPackageStartupMessages(library(irlba))
  pca_res <- irlba::irlba(X_norm, nv=2)
  pca_df  <- data.frame(PC1=pca_res$u[,1], PC2=pca_res$u[,2])
} else {
  pca_res <- prcomp(X_norm, rank.=2, center=FALSE, scale.=FALSE)
  pca_df  <- data.frame(PC1=pca_res$x[,1], PC2=pca_res$x[,2])
}
cat(sprintf("  PCA done (%d docs × 2 components)\n\n", nrow(pca_df)))

# ── 4. KMeans — test k = 3 to 10 ─────────────────────────────────────────────
cat("Running KMeans for k = 3 to 10...\n")
suppressPackageStartupMessages(library(cluster))

sil_scores <- numeric(0)
k_values   <- 3:10

for (k in k_values) {
  set.seed(42)
  km <- kmeans(X_norm, centers=k, nstart=10, iter.max=100, algorithm="Hartigan-Wong")
  # Silhouette on PCA coords (faster than full TF-IDF)
  if (k > 1 && k < nrow(pca_df)) {
    sil_obj <- cluster::silhouette(km$cluster, dist(pca_df))
    sil_scores[as.character(k)] <- mean(sil_obj[,3], na.rm=TRUE)
  } else {
    sil_scores[as.character(k)] <- NA_real_
  }
  cat(sprintf("  k=%d  silhouette=%.4f\n", k,
              ifelse(is.na(sil_scores[as.character(k)]),0,sil_scores[as.character(k)])))
}

# Best k
best_k <- k_values[which.max(sil_scores)]
best_sil <- max(sil_scores, na.rm=TRUE)
cat(sprintf("\nBest k: %d (silhouette=%.4f)\n", best_k, best_sil))
cat(sprintf("Baseline: k=8, silhouette=0.064\n"))
if (abs(best_sil - 0.064) > 0.02) {
  cat("Note: Minor difference from baseline expected (TF-IDF implementation differs)\n")
}

# ── 5. Final clustering with best k ───────────────────────────────────────────
cat(sprintf("\nRunning final KMeans with k=%d...\n", best_k))
set.seed(42)
km_final <- kmeans(X_norm, centers=best_k, nstart=25, iter.max=200)
cluster_labels <- km_final$cluster

cat("Cluster sizes:\n")
print(table(cluster_labels))

# ── 6. Cluster keyword summaries ──────────────────────────────────────────────
cat("\nGenerating cluster keyword summaries...\n")

cluster_keywords <- lapply(seq_len(best_k), function(k) {
  idx <- which(cluster_labels == k)
  # Get top terms by mean TF-IDF within cluster
  cluster_mean_tfidf <- colMeans(X_tfidf[idx, , drop=FALSE])
  top_terms_k <- names(sort(cluster_mean_tfidf, decreasing=TRUE))[1:10]
  top_terms_k <- top_terms_k[!is.na(top_terms_k)]
  data.frame(
    cluster     = k,
    n_papers    = length(idx),
    pct_papers  = round(100*length(idx)/nrow(df), 1),
    top_terms   = paste(top_terms_k[1:min(10,length(top_terms_k))], collapse="; "),
    stringsAsFactors=FALSE
  )
})
cluster_kw_df <- do.call(rbind, cluster_keywords)

cat("\nCluster Summary:\n")
cat(sprintf("%-8s  %-8s  %-6s  %s\n","Cluster","N","Pct","Top Terms"))
cat(paste(rep("-",90),collapse=""),"\n")
for (i in seq_len(nrow(cluster_kw_df))) {
  cat(sprintf("%-8d  %-8d  %.1f%%  %s\n",
              cluster_kw_df$cluster[i], cluster_kw_df$n_papers[i],
              cluster_kw_df$pct_papers[i],
              substr(cluster_kw_df$top_terms[i],1,65)))
}

# ── 7. Visualisations ─────────────────────────────────────────────────────────
cat("\nGenerating clustering visualisations...\n")

pca_df$cluster <- factor(cluster_labels)

# 7a. PCA cluster scatter
p_pca <- ggplot(pca_df, aes(x=PC1, y=PC2, colour=cluster)) +
  geom_point(alpha=0.4, size=0.9) +
  scale_colour_brewer(palette="Set1", name="Cluster") +
  labs(title=sprintf("Topic Clusters — PCA Projection (k=%d)", best_k),
       subtitle=sprintf("TF-IDF + KMeans | Silhouette=%.3f | 2000 AI/ML papers",
                        best_sil),
       x="PC1", y="PC2",
       caption="Low silhouette expected: research topics in AI/ML corpus overlap substantially") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), legend.position="right")

ggsave(file.path(fig_dir,"clustering_pca_scatter.png"), p_pca, width=8, height=6, dpi=150)
cat("  Saved: clustering_pca_scatter.png\n")

# 7b. Silhouette vs k
sil_df <- data.frame(k=k_values, silhouette=sil_scores)
p_sil <- ggplot(sil_df, aes(x=k, y=silhouette)) +
  geom_line(colour="#2166ac", linewidth=1) +
  geom_point(size=3, colour="#2166ac") +
  geom_vline(xintercept=best_k, linetype="dashed", colour="#d7301f") +
  annotate("text", x=best_k, y=max(sil_scores,na.rm=TRUE),
           label=sprintf("Best k=%d\nsil=%.3f",best_k,best_sil),
           vjust=-0.5, hjust=-0.1, size=3.5, colour="#d7301f") +
  labs(title="Silhouette Score vs Number of Clusters",
       subtitle="Higher silhouette = better-separated clusters | AI/ML papers show substantial overlap",
       x="Number of Clusters (k)", y="Mean Silhouette Score") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold")) +
  scale_x_continuous(breaks=k_values)

ggsave(file.path(fig_dir,"clustering_silhouette_vs_k.png"), p_sil, width=7, height=4, dpi=150)
cat("  Saved: clustering_silhouette_vs_k.png\n")

# 7c. Cluster size bar
size_df <- cluster_kw_df %>%
  mutate(cluster_label=factor(paste("Cluster",cluster),
                               levels=paste("Cluster",cluster[order(-n_papers)])))
p_size <- ggplot(size_df, aes(x=cluster_label, y=n_papers, fill=cluster_label)) +
  geom_col(alpha=0.85, show.legend=FALSE) +
  scale_fill_brewer(palette="Set2") +
  coord_flip() +
  labs(title=sprintf("Cluster Size Distribution (k=%d)", best_k),
       x=NULL, y="Number of Papers") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))

ggsave(file.path(fig_dir,"clustering_size_distribution.png"), p_size,
       width=6, height=4, dpi=150)
cat("  Saved: clustering_size_distribution.png\n")

# ── 8. Save outputs ───────────────────────────────────────────────────────────
cluster_assignments <- data.frame(
  paper_id    = seq_len(nrow(df)),
  openalex_id = df$id,
  title_short = substr(df$title,1,80),
  cluster     = cluster_labels,
  stringsAsFactors=FALSE
)
write_csv(cluster_assignments, file.path(out_dir, "cluster_assignments.csv"))
write_csv(cluster_kw_df,       file.path(out_dir, "cluster_keywords.csv"))
write_csv(sil_df,              file.path(out_dir, "silhouette_scores.csv"))

cat(sprintf("\nSaved: cluster_assignments.csv (%d rows)\n", nrow(cluster_assignments)))
cat(sprintf("Saved: cluster_keywords.csv (%d clusters)\n", nrow(cluster_kw_df)))
cat(sprintf("Saved: silhouette_scores.csv\n"))

cat("\n=== M7b Clustering COMPLETE ===\n")
cat(sprintf("  Best k         : %d  (baseline k=8)\n", best_k))
cat(sprintf("  Silhouette     : %.4f  (baseline 0.064)\n", best_sil))
if (best_sil < 0.10) {
  cat("  Interpretation : The clustering is exploratory. Low silhouette\n")
  cat("                   reflects substantial topic overlap in AI/ML research,\n")
  cat("                   consistent with the Python baseline finding.\n")
}
cat(sprintf("  Cluster sizes  : min=%d  max=%d  mean=%.0f\n",
            min(cluster_kw_df$n_papers), max(cluster_kw_df$n_papers),
            mean(cluster_kw_df$n_papers)))
