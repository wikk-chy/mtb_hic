setwd('D:/BaiduNetdiskDownload/c')
op<-read.csv('operon_list_updated_with_bin_coverage.csv')
int<-read.csv('operon_interaction_matrix_log.csv',row.names = 1)
random_all<-data.frame()
for (z in 1:4) {
random<-data.frame()
random[1,1]<-'Rv0007_up'
i=10196
m=1
y=2
t=0

while (i<4411532) {
  x=runif(1,min=0,max=1)
  current_seed <- .Random.seed
  .Random.seed<-NULL
  if (x<0.5) {
    i=i+70000+(0-x)*2*10000
  }else{i=i+70000+(x-0.5)*2*10000}
  for (j in m:749) {
    if (op[j,11]<i&op[j,12]>i) {
      random[y,1]=op[j,1]
      m =j
      t=1
      break
      }
    
    }
  if (t==0) {for (j in m:749) {
    if (j==749) {
      break
    }
    if (op[j,12]<i&op[j+1,11]>i) {
      random[y,1]=op[j+1,1]
      m =j
      t=1
      break} 
  }
  }
  t=0
  y=y+1
}
a<-data.frame(random[seq(1,nrow(random),2),])  
b<-data.frame(random[seq(0,nrow(random),2),])   
if(nrow(a) != nrow(b)){a<-data.frame(a[-nrow(a),])}
pair<-cbind(a,b)
random <- pair[pair[,1] != pair[,2], ]             
for (i in 1:nrow(random)) {
  random[i,3]<-int[random[i,1],random[i,2]]
}
filter<-read.csv('filtered_operon_pairs.csv')
for (i in 1:nrow(random)) {
  random[i,4]<-min(op[which(op[,1] == random[i,1]),16],op[which(op[,1] == random[i,2]),16])/max(op[which(op[,1] == random[i,1]),16],op[which(op[,1] == random[i,2]),16])
}
colnames(random)[1]='op1'
colnames(random)[2]='op2'
colnames(random)[3]='interaction_strength'
colnames(random)[4]='expression_similarity'
if (ncol(random_all) == 0) {
  random_all<-random
} else {
  random_all<-rbind(random_all,random)}
}
library(ggplot2)
library(dplyr)
df1 <- random_all %>% mutate(group = "random")
df2 <- filter %>% mutate(group = "filter")
# 合并数据框
combined_df <- bind_rows(df1, df2)

# 绘制带图例的散点图
centers <- combined_df %>%
  group_by(group) %>%
  summarise(
    mean_x = mean(interaction_strength),
    mean_y = mean(expression_similarity)
  )
ggplot(combined_df, aes(x = combined_df$interaction_strength, y = combined_df$expression_similarity, color = group,fill = group)) +
  geom_point(size = 1.5) +
  scale_color_manual(values = c("random" = "#4D779B", "filter" = "#AE3019")) +
  labs(x = "interaction_strength", y = "expression_similarity", color = "group",title = current_seed) +
  theme(
    panel.background = element_rect(fill = "white", colour = "black", linewidth = 1), # 白色背景，黑色边框
    legend.position = "right"            # 图例位置
  ) +stat_ellipse(
    geom = "polygon",  # 使用填充多边形
    alpha = 0.05,       # 透明度
    level = 0.9,       # 90%置信区间
    linewidth = 1      # 边框线宽
  ) +  geom_point(
    data = centers,
    aes(x = mean_x, y = mean_y),
    shape = 23,        # 圆形带边框
    size = 2,          # 点大小
    color = "black",   # 边框颜色
    fill = "white",    # 填充颜色
    stroke = 1.5       # 边框线宽
  ) +coord_fixed(ratio = 0.9,xlim = c(3.45, NA))+
  scale_color_manual(values = c("random" = "#4D779B", "filter" = "#AE3019")) +
  scale_fill_manual(values = c("random" = "#4D779B", "filter" = "#AE3019"))+
  theme(legend.position = "right")
write.csv(random_all,'random_pairs.csv')
