########################################
# 1) Caricamento librerie
########################################
library(readxl)
library(sf)
library(dplyr)
library(spdep)
library(ggplot2)
library(cluster)
library(factoextra)
library(fpc)
library(dbscan)
library(tidyr)
library(viridis)
library(ggpubr)
library(igraph)
library(scales)
library(gridExtra)
library(MASS)
library(spatstat)
library(knitr)
library(rmarkdown)

########################################
# 2) Lettura da Excel
########################################
# Modifica il path in base alla tua struttura di cartelle
excel_file <- "E:\\INGV\\1_Human Mobility\\2_FindCircular_Algorithm\\R&RandomForerst\\OmanData.xlsx"

# Supponendo che il tuo foglio abbia intestazioni e una colonna "ID" 
# e le colonne: "Diameter","Width","Lenght","Height","Long","Lat"
df_oman <- read_excel(excel_file, sheet = 1)

# Converti Diameter, Width, Lenght, Height in numerico
df_oman <- df_oman %>%
  mutate(across(c(Diameter, Width, Lenght, Height), ~as.numeric(as.character(.))))

# Converte in oggetto sf usando le colonne Long/Lat
# Se conosci l'EPSG corretta, sostituisci "4326" con quello appropriato
df_oman_sf <- st_as_sf(df_oman, 
                       coords = c("Long","Lat"), 
                       crs = 4326,    # <--- imposta EPSG corretto
                       remove = FALSE)

########################################
# 3) Selezione colonne di interesse
########################################
columns_of_interest <- c("ID","Diameter","Width","Lenght","Height","Long","Lat")
subset_data <- df_oman_sf %>% 
  dplyr::select(any_of(columns_of_interest), geometry)

# ✅ Convertiamo in numerico
subset_data <- subset_data %>%
  mutate(across(c(Diameter, Width, Lenght, Height), as.numeric))

print(names(subset_data))
print(head(subset_data))

########################################
# 4) Gestione dei valori mancanti
########################################
# Sostituisce eventuali NA numerici con la mediana
subset_data <- subset_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

########################################
# 5) Statistiche descrittive generali
########################################
# summary() restituisce un oggetto di tipo "table". 
# Per scriverlo in CSV in modo "testuale" usiamo sink() + cat/print:
general_stats <- summary(st_drop_geometry(subset_data))
sink("gen_stats_results.csv")
cat("General Stats (Summary)\n")
print(general_stats)
sink()

########################################
# 6) Creazione lista di vicinato (k=10 nearest neighbors)
########################################
coords <- st_coordinates(subset_data) # matrice x/y
k <- 10
nb_knn <- knn2nb(knearneigh(coords, k = k))
listw <- nb2listw(nb_knn, style = "W")

# Salviamo le coordinate X/Y in subset_data per comodità
subset_data$X <- coords[,1]
subset_data$Y <- coords[,2]

########################################
# 7) Moran’s I Test globale
########################################
moran_test_Diam <- moran.test(subset_data$Diameter, listw)
moran_test_Len  <- moran.test(subset_data$Lenght,  listw)

# Funzione di utilità per convertire i risultati in dataframe
test2df <- function(mtest_obj, test_name="Moran") {
  data.frame(
    method      = mtest_obj$method,
    statistic   = mtest_obj$statistic,
    estimate    = as.numeric(mtest_obj$estimate),
    p.value     = mtest_obj$p.value,
    alternative = mtest_obj$alternative,
    test_name   = test_name
  )
}

moran_diam_df <- test2df(moran_test_Diam, "Moran_Diameter")
moran_len_df  <- test2df(moran_test_Len,  "Moran_Length")
write.csv2(moran_diam_df, "moran_diam_results.csv", row.names = FALSE)
write.csv2(moran_len_df,  "moran_len_results.csv", row.names = FALSE)

########################################
# 8) Local Moran’s I (LISA) su Diameter
########################################
local_moran <- localmoran(subset_data$Diameter, listw)
local_moran_df <- as.data.frame(local_moran)

# Aggiungiamo la colonna ID corrispondente
local_moran_df$ID <- subset_data$ID

write.csv2(local_moran_df, "local_moran_df_results.csv", row.names = FALSE)

########################################
# 9) Plot Moran Scatter (opzionale)
########################################
moran_I <- moran_test_Diam$estimate["Moran I"]
p_value <- moran_test_Diam$p.value

moran_scatter_df <- data.frame(
  Z  = scale(subset_data$Diameter),
  WZ = scale(lag.listw(listw, subset_data$Diameter))
)

moran_scatter_plot <- ggplot(moran_scatter_df, aes(x = Z, y = WZ)) +
  geom_point(alpha = 0.7, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray") +
  labs(title = paste("Moran Scatter Plot (I =", round(moran_I, 3), ")"),
       subtitle = paste("p-value:", round(p_value, 5)),
       x = "Standardized Diameter",
       y = "Spatially Lagged Diameter") +
  theme_minimal()
ggsave("moran_scatter_plot.png", plot = moran_scatter_plot)

########################################
# 10) Geary’s C Test
########################################
geary_test_diam <- geary.test(subset_data$Diameter, listw)
geary_test_len  <- geary.test(subset_data$Lenght,  listw)

geary_diam_df <- test2df(geary_test_diam, "Geary_Diameter")
geary_len_df  <- test2df(geary_test_len,  "Geary_Length")
write.csv2(geary_diam_df, "geary_diam_results.csv", row.names = FALSE)
write.csv2(geary_len_df,  "geary_len_results.csv", row.names = FALSE)

########################################
# 11) K-Means Clustering
########################################
# Usiamo X/Y standardizzati
clustering_data <- subset_data %>% 
  st_drop_geometry() %>%
  dplyr::select(X, Y)

# Standardizziamo i dati
clustering_data_scaled <- scale(clustering_data)

# Elbow Method
elbow_plot <- fviz_nbclust(clustering_data_scaled, kmeans, method = "wss") +
  labs(title = "Elbow Method for Optimal Clusters")
ggsave("elbow_plot.png", plot = elbow_plot)

# Silhouette
silhouette_plot <- fviz_nbclust(clustering_data_scaled, kmeans, method = "silhouette") +
  labs(title = "Silhouette Method for Optimal Clusters")
ggsave("silhouette_plot.png", plot = silhouette_plot)

# Esegui K-Means
set.seed(123)
optimal_k <- 4  # scelto come esempio
# 1) Creiamo un data frame da clustering_data_scaled
clustering_data_df <- as.data.frame(clustering_data_scaled)

# 2) Applichiamo kmeans su tale data frame
kmeans_result <- kmeans(clustering_data_df, centers = optimal_k, nstart = 10)

# 3) Salviamo i cluster in un df con ID
kmeans_clusters_df <- data.frame(
  ID      = subset_data$ID,
  cluster = kmeans_result$cluster
)
write.csv2(kmeans_clusters_df, "kmeans_results.csv", row.names = FALSE)

# Visualizzazione
kmeans_plot <- fviz_cluster(kmeans_result, data = clustering_data_df) +
  labs(title = "K-Means Clustering")
ggsave("kmeans_plot.png", plot = kmeans_plot)

########################################
# 12) DBSCAN Clustering
########################################
set.seed(123)
# DBSCAN lavora su un data frame / matrice numerica
dbscan_result <- dbscan(clustering_data_df, eps = 0.5, MinPts = 2)

# Prepariamo un dataframe cluster+ID
dbscan_clusters_df <- data.frame(
  ID      = subset_data$ID,
  cluster = dbscan_result$cluster
)
write.csv2(dbscan_clusters_df, "dbscan_results.csv", row.names = FALSE)

# Esempio di plot per DBSCAN
dbscan_plot <- ggplot(clustering_data_df, 
                      aes(x = X, y = Y, color = factor(dbscan_result$cluster))) +
  geom_point() +
  theme_minimal() +
  labs(title = "DBSCAN Clustering", color = "Cluster")
ggsave("dbscan_plot.png", plot = dbscan_plot)

########################################
# 13) Nearest Neighbor Distance
########################################
nnd <- nndist(coords) # da spatstat
mean_nnd <- mean(nnd)
sd_nnd   <- sd(nnd)

nn_df <- data.frame(
  MeanNND = mean_nnd,
  SdNND   = sd_nnd
)
write.csv2(nn_df, "nearest_neighbor_results.csv", row.names = FALSE)

########################################
# 14) Hotspot Analysis (Gi*)
########################################
gi_weights <- nb2listw(nb_knn, style = "W")
gi_star <- localG(subset_data$Diameter, gi_weights)

# ✅ Converti in vettore numerico e gestisci NA o infiniti
gi_star <- as.numeric(gi_star)

# ✅ Controlla eventuali problemi (NA o Inf)
print(summary(gi_star))

# ✅ Sostituisci eventuali NA o infiniti con 0
gi_star[is.na(gi_star) | is.infinite(gi_star)] <- 0

# ✅ Normalizza per visualizzazione migliore
subset_data$Gi_star <- scales::rescale(gi_star, to = c(-1, 1))

# ✅ Salvataggio del risultato in CSV
hotspot_df <- data.frame(
  ID      = subset_data$ID,
  Gi_star = gi_star
)
write.csv2(hotspot_df, "hotspot_results.csv", row.names = FALSE)

# ✅ Creiamo il plot corretto
hotspot_plot <- ggplot(subset_data, aes(x = X, y = Y, color = gi_star)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Hotspot Analysis (Gi*)", color = "Gi* z-score")

ggsave("hotspot_plot.png", plot = hotspot_plot)


########################################
# 15) Kernel Density (opzionale)
########################################
kde <- MASS::kde2d(coords[,1], coords[,2], n=100)
kde_df <- expand_grid(x = kde$x, y = kde$y)
kde_df$z <- as.vector(kde$z)

kde_plot <- ggplot(kde_df, aes(x = x, y = y, fill = z)) +
  geom_tile() +
  scale_fill_viridis_c(option = "C") +
  labs(title = "Kernel Density Estimation", fill = "Density") +
  theme_minimal()
ggsave("kde_plot.png", plot = kde_plot)

########################################
# 16) Render di un RMarkdown (opzionale)
########################################
# Se hai "Spatial Statistics Report.Rmd" nella stessa cartella
# rmarkdown::render("Spatial Statistics Report.Rmd", output_format = "pdf_document")

########################################
# Fine
########################################
