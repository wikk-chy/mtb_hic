setwd('D:/BaiduNetdiskDownload/c')
library(stringr)
a<-read.csv('full_edges.csv')
b<-read.csv('operon_edges.csv')
operon <- subset(b, b[,3] == "operon-operon")
op<-sapply(split(operon[,1], operon[,2]), function(x) paste(unique(x), collapse = ", "))
pattern_op <- paste(op, collapse = "|")
bind<-subset(a, a[,3] == "DNA-Protein")
bind_list<-unique(bind$source)
bin_size <- 5000
loop<-read.table('wt-loops.bedpe',header = F)
loop[,2] <- paste0("bin", ceiling(loop[,2] / bin_size))
loop[,3] <- paste0("bin", ceiling(loop[,3] / bin_size))
loop[,5] <- paste0("bin", ceiling(loop[,5] / bin_size))
loop[,6] <- paste0("bin", ceiling(loop[,6] / bin_size))
cid<-read.table('wt-boundary.bed',header = F)
cid[,2] <- paste0("bin", ceiling(cid[,2] / bin_size+1))
cid[,3] <- paste0("bin", ceiling(cid[,3] / bin_size))
sum_op<-0
sum_loop<-0
sum_cid<-0
sum_pro<-0
for (i in bind_list) {
  df <- subset(a, a[,1] == i)
  df1 <- subset(b, b[,1] == i)
  protein<-subset(df, df[,3] == "Protein-Protein")
  sum_pro<-sum_pro+nrow(protein)
  matching_rows <- apply(df1[, 1:2], 1, function(row) {
    any(str_detect(row, regex(pattern_op, ignore_case = TRUE)))
  })
  sum_op <- sum_op+sum(matching_rows) 
  reference_values <- df[, 2]
  matchingloop <- which(loop[, 3] %in% reference_values & loop[, 6] %in% reference_values)
  sum_loop <- sum_loop +length(matchingloop)
  matchingcid<- which(cid[, 2] %in% reference_values | cid[, 3] %in% reference_values)
  sum_cid <- sum_cid +length(matchingcid)
  }
avg_pro<-round(sum_pro/length(bind_list),2)
avg_cid<-round(sum_cid/length(bind_list),2)
avg_loop<-round(sum_loop/length(bind_list),2)
avg_op<-round(sum_op/length(bind_list),2)

library(ggplot2)
library(tidyr)
library(dplyr)

# 创建数据
data <- data.frame(
  Group = c("protein", "operon", "cid", "loop"),
  Count = c(14, 7, 21, 10),
  Mean = c(avg_pro, avg_op, 3.65,avg_cid)
)

# 转换数据为长格式
data_long <- data %>%
  pivot_longer(cols = c(Count, Mean), 
               names_to = "Type", 
               values_to = "Value")
windowsFonts(A=windowsFont('Arial'))
# 绘制分组柱状图
ggplot(data_long, aes(x = Group, y = Value, fill = Type)) +
  geom_col(position = "dodge", width = 0.7, color = "black", size = 0.5) +
  geom_text(aes(label = round(Value, 2)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Count" = "#4E79A7", "Mean" = "#F28E2B"),
                    labels = c("Count" = "count", "Mean" = "average")) +
  labs(x = "", y = "count", fill = "type",size = 16) +
  theme_classic() +theme(text = element_text(family = 'A',face = 'bold'))
  theme(panel.border = element_rect(colour = "black", fill = NA, size = 1),
    axis.ticks.x = element_blank(), 
    axis.text = element_text(size = 11, color = "black"),
    legend.position = "top"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
