setwd('D:/新建文件夹/20241027结核3D基因组')
library(ggplot2)
library(dplyr)
library(ggrepel)
a<-read.csv('full_nodes.csv')
b<-read.csv('full_edges.csv')
type_counts <- b %>%
  group_by(interaction_type) %>%
  summarise(count = n())
ggplot(type_counts, aes(x = interaction_type, y = count, fill = interaction_type)) +
  geom_bar(stat = "identity") +  # 添加黑色边框
  geom_text(aes(label = count), hjust = -0.2, size = 5) +   # 在条形上方显示数量
  labs(x = "Type", 
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("DNA-DNA" = "#7E99F4", "Protein-Protein" = "#CC7C71","DNA-Protein"="#7AB656")) +
  ylim(0, max(type_counts$count) * 1.2)+

  coord_flip()+
  theme(
    axis.ticks.length = unit(0.2,'cm'),
    axis.ticks = element_line(size = 1),
    legend.position = "none",  # 移除图例
    panel.grid.major = element_blank(),  # 去除主要网格线
    panel.grid.minor = element_blank(), # 去除次要网格线
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)  
    ) 
