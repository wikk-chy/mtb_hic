setwd('D:/mtb/新建文件夹/3')
library(stringi)
library(ggplot2)
library(dplyr)
downgokegg<-read.delim("lakegg.txt")
enrich<-downgokegg
enrich_signif=enrich[which(enrich$PValue<0.05),]
enrich_signif=enrich_signif[,c(1:3,5)]
head(enrich_signif)
enrich_signif=data.frame(enrich_signif)
KEGG=enrich_signif
KEGG$Term<-stri_sub(KEGG$Term,10,100)
KEGG$PValue<--log10(KEGG$PValue)  
ggplot(KEGG, aes(x = Count, y = reorder(Term, Count), fill = PValue)) +
  geom_bar(stat = "identity", width = 0.8) +
  geom_text(aes(label = sprintf("P=%.3g", PValue)), 
            hjust = -0.1, size = 3.5, color = "black") +
  scale_fill_gradient(low = "#A8CAE8", high = "#F58383", name = "P-value") +
  theme_bw() +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.text.y = element_text(size = 11),
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 13, face = "bold")
  ) +
  xlab("Gene Count") +
  ylab("KEGG Pathway") +
  ggtitle("KEGG Pathway Enrichment (colored by P-value)") +
  coord_cartesian(xlim = c(0, max(KEGG$Count) * 1.3))
GO_CC<-read.delim('lacc.txt')
GO_CC<-data.frame()
GO_CC_signif=GO_CC[which(GO_CC$PValue<0.1),]
GO_CC_signif=GO_CC[,c(1:3,5)]
head(GO_CC_signif)
GO_CC_signif=data.frame(GO_CC_signif)
GO_CC_signif$Term<-stri_sub(GO_CC_signif$Term,12,100)
GO_BP<-read.delim('labp.txt')
GO_BP_signif=GO_BP[which(GO_BP$PValue<0.1),]
GO_BP_signif=GO_BP_signif[,c(1:3,5)]
head(GO_BP_signif)
GO_BP_signif=data.frame(GO_BP_signif)
GO_BP_signif$Term<-stri_sub(GO_BP_signif$Term,12,100)
GO_MF<-read.delim('lamf.txt')
GO_MF_signif=GO_MF[which(GO_MF$PValue<0.1),]
GO_MF_signif=GO_MF_signif[,c(1:3,5)]
head(GO_MF_signif)
GO_MF_signif=data.frame(GO_MF_signif)
GO_MF_signif$Term<-stri_sub(GO_MF_signif$Term,12,100)
enrich_signif=rbind(GO_BP_signif,rbind(GO_CC_signif,GO_MF_signif))
go=enrich_signif
go=arrange(go,go$Category,go$PValue)
m=go$Category
m=gsub("TERM","",m)
m=gsub("_DIRECT","",m)
go$Category=m
GO_term_order=factor(as.integer(rownames(go)),labels = go$Term)
COLS<-c("#66C3A5","#8DA1CB","#FD8D62")                                          
ggplot(data=go,aes(x=GO_term_order,y=Count,fill=Category))+
  geom_bar(stat = "identity",width = 0.8)+
  scale_fill_manual(values = COLS)+
  theme_bw()+
  xlab("Terms")+
  ylab("Gene_counts")+
  labs()+coord_flip()+
  theme(axis.text.x = element_text(face = "bold",color = "black",angle = 90,vjust = 1,hjust = 1)) 
library(dplyr)

go <- go %>%
  mutate(
    Category = factor(Category, 
                      levels = c("GO_BP","GO_MF","GO_CC"),
                      labels = c("BP","MF","CC")),
    logP = -log10(PValue)             # 颜色用这个
  ) %>%
  arrange(Category, logP) %>%         # 先按 BP/MF/CC，再按显著性排
  mutate(GO_term_order = factor(GO_term_order, 
                                levels = unique(GO_term_order)))
ggplot(data = go,
       aes(x = GO_term_order, y = Count, fill = logP)) +
  geom_bar(stat = "identity", width = 0.8) +
  # 颜色用 P 值渐变
  scale_fill_gradient(name = "-log10(P)",
                      low  = "#A8CAE8",
                      high = "#F58383") +
  # 按 BP / MF / CC 分面分组
  facet_grid(Category ~ ., scales = "free_y", space = "free_y") +
  theme_bw() +
  xlab("Terms") +
  ylab("Gene_counts") +
  coord_flip() +
  theme(
    axis.text.y = element_text( color = "black"),
    legend.position = "right"
  )
