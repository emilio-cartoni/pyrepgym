require(data.table)

df = fread("data_explore")
names(df) <- c(
    mapply(function(x){paste('j', x, sep='')}, seq(7)),
    c('x', 'y', 'z'),
    c('qi', 'qj','qk','w')
)

ddf <- df[df$z>=0.4 & df$z<=0.7]
fwrite(ddf, "data_filtered", sep = " ", 
       col.names = F)




