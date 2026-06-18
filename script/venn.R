library(ggVennDiagram)
library(VennDiagram)
setwd('D:/hic/新建文件夹')
hy<-read.csv('hypoxiaWTvsWT_supo_DEG_limma.csv')
la<-read.csv('LatentvsWT_supo_DEG_limma.csv')
ko<-read.csv('KOvsWT_supo_DEG_limma.csv')
kohy<-read.csv('KOhyvsWT_supo_DEG_limma.csv')
la <- subset(la, la[,4] != "Stable")
ko <- subset(ko, ko[,4] != "Stable")
kohy <- subset(kohy, kohy[,4] != "Stable")

set_hy<-hy[,1]
set_la<-la[,1]
set_ko<-ko[,1]
set_kohy<-kohy[,1]
# 创建列表数据
my_list <- list(
  hy = set_hy,
  la = set_la,
  ko = set_ko,
  kohy = set_kohy
)

venn.plot <- venn.diagram(
  x = my_list,
  filename = NULL,  # 不保存到文件，直接显示
  fill = c("#8ab07c", "#f3a361", "#299d8f", "#e7c66b"),
  alpha = 0.5,
  label.col = "black",
  euler.d = TRUE,  # 启用欧拉图模式，使圆更紧凑
  scaled = TRUE,
  fontfamily = "serif",
  fontface = "bold",
  cat.col = c("#8ab07c", "#f3a361", "#299d8f", "#e7c66b"),
  cat.cex = 1.5,
  cat.fontfamily = "serif"
)
grid.draw(venn.plot)
ggsave('vennplot.pdf',venn.plot)
BBcommon_elements <- Reduce(intersect, my_list)
print(common_elements)
write.csv(common_elements,'genlist.csv')
