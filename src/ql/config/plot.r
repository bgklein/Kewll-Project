library("ggplot2")
library("reshape2")


plot_single = function(){
	filename = "default"

	data_df = read.csv(paste(filename, '.csv', sep=''))

	p1 = ggplot(data=data_df, aes(x=epoch, y=score)) +
		    geom_point() + 
		    geom_line() +
		    ylim(c(0, 500)) + 
		    xlab("Epoch") +
		    ylab("Accuracy") +
		    ggtitle("Rewards vs epoch")

	ggsave(paste(filename, ".png", sep = ""), width=6, height=4, dpi=300)
}

plot_mini = function(){
	df1 = read.csv('m_batch_size=16.csv', sep=',')
	df1$m = rep(16, dim(df1)[1])
	df2 = read.csv('m_batch_size=8.csv', sep=',')
	df2$m = rep(8, dim(df2)[1])
	df3 = read.csv('m_batch_size=64.csv', sep=',')
	df3$m = rep(64, dim(df3)[1])
	df4 = read.csv('m_default.csv', sep=',')
	df4$m = rep(32, dim(df4)[1])

	tables = c(df1, df2, df3, df4)

	df_total = merge(merge(merge(df1, df2, all=T), df3, all=T), df4, all=T)
	df_total$minibatch_size = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=times, y=score, 
				group=minibatch_size, color=minibatch_size)) +
		    geom_point(size = 0.7) + 
		    geom_line(size = 0.3) +
    		# stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Second") +
		    ylab("Mean Score") +
		    ggtitle("Mean Score v.s. Time with Different Minibatch Size") +
		    scale_fill_discrete(name = "Minibatch size")
    ggsave(paste("./plots/minibatch.png", sep = ""), width=10, height=6, dpi=300)
}


plot_alpha = function(){
	df1 = read.csv('m_alpha=0.5.csv', sep=',')
	df1$m = rep(0.5, dim(df1)[1])
	df2 = read.csv('m_alpha=0.7.csv', sep=',')
	df2$m = rep(0.7, dim(df2)[1])
	df3 = read.csv('m_alpha=0.9.csv', sep=',')
	df3$m = rep(0.9, dim(df3)[1])
	df4 = read.csv('m_default.csv', sep=',')
	df4$m = rep(1.0, dim(df4)[1])

	tables = c(df1, df2, df3, df4)

	df_total = merge(merge(merge(df1, df2, all=T), df3, all=T), df4, all=T)
	df_total$alpha = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=epoch, y=score, 
				group=alpha, color=alpha)) +
		    # geom_point() + 
		    geom_line(size=0.5) +
    		# stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Epoch") +
		    ylab("Mean Score") +
		    ggtitle("Mean Score v.s. Epoch with Different Alpha") +
    ggsave(paste("./plots/alpha_smooth.png", sep = ""), width=10, height=6, dpi=300)
}

plot_eta = function(){
	df1 = read.csv('m_eta=5e-05.csv', sep=',')
	df1$m = rep(0.00005, dim(df1)[1])
	df2 = read.csv('m_default.csv', sep=',')
	df2$m = rep(0.001, dim(df2)[1])
	df3 = read.csv('m_eta=0.005.csv', sep=',')
	df3$m = rep(0.005, dim(df3)[1])

	tables = c(df1, df2, df3)

	df_total = merge(merge(df1, df2, all=T), df3, all=T)
	df_total$eta = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=epoch, y=score, 
				group=eta, color=eta)) +
		    geom_point() + 
		    # geom_line(size=0.3) +
    		stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Epoch") +
		    ylab("Mean Score") +
		    ggtitle("Score v.s. Epoch with Different Learning Rate") +
    ggsave(paste("./plots/eta_smooth.png", sep = ""), width=10, height=6, dpi=300)
}

plot_msize = function(){
	df1 = read.csv('m_memory_size=1000.csv', sep=',')
	df1$m = rep(1000, dim(df1)[1])
	df2 = read.csv('m_memory_size=3000.csv', sep=',')
	df2$m = rep(3000, dim(df2)[1])
	df3 = read.csv('m_memory_size=5000.csv', sep=',')
	df3$m = rep(4000, dim(df3)[1])
	df4 = read.csv('m_memory_size=10000.csv', sep=',')
	df4$m = rep(10000, dim(df4)[1])
	df5 = read.csv('m_default.csv', sep=',')
	df5$m = rep(2000, dim(df5)[1])
	df6 = read.csv('m_memory_size=500.csv', sep=',')
	df6$m = rep(500, dim(df6)[1])

	tables = c(df1, df2, df3, df4, df5, df6)

	df_total = merge(merge(merge(merge(merge(df1, df2, all=T), df3, all=T), 
					 	   df4, all=T),
					 df5, all=T), df6, all=T)


	df_total$memory_size = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=epoch, y=score, 
				group=memory_size, color=memory_size)) +
		    geom_point() + 
		    geom_line(size=0.3) +
    		# stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Epoch") +
		    ylab("Mean Score") +
		    ggtitle("Mean Score v.s. Epoch with Different Memory Szie") +
    ggsave(paste("./plots/memory_size.png", sep = ""), 
    		width=10, height=6, dpi=300)
}

plot_p = function(){
	df1 = read.csv('m_policy.csv', sep=',')
	df1$m = rep('policy-based', dim(df1)[1])
	df2 = read.csv('m_default.csv', sep=',')
	df2$m = rep('q-learning', dim(df2)[1])

	tables = c(df1, df2)

	df_total = merge(df1, df2, all=T)

	df_total$method = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=times, y=score, 
				group=method, color=method)) +
		    geom_point() + 
		    geom_line(size=0.3) +
    		stat_smooth(se = FALSE, size=1) +
		    ylim(c(0, 500)) + 
		    xlab("Second") +
		    ylab("Mean Score") +
		    ggtitle("Q-learning v.s. policy-based Gradient") +
    ggsave(paste("./plots/qvsp_smooth.png", sep = ""), width=10, height=6, dpi=300)
}

plot_e = function(){
	df1 = read.csv('m_elu.csv', sep=',')
	df1$m = rep('ELU', dim(df1)[1])
	df2 = read.csv('m_default.csv', sep=',')
	df2$m = rep('ReLU', dim(df2)[1])

	tables = c(df1, df2)

	df_total = merge(df1, df2, all=T)

	df_total$activation = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=times, y=score, 
				group=activation, color=activation)) +
		    geom_point(size=0.7) + 
		    geom_line(size=0.3) +
    		stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Second") +
		    ylab("Mean Score") +
		    ggtitle("ReLU v.s. ELU") +
    ggsave(paste("./plots/rvse_smooth.png", sep = ""), width=10, height=6, dpi=300)
}

plot_nn = function(){
	df1 = read.csv('nn.csv', sep=',')
	colnames(df1) = c('epoch', 'score', 'times')
	df1$m = rep('Traditional DNN', dim(df1)[1])
	df2 = read.csv('m_default.csv', sep=',')
	df2$m = rep('Q-learning', dim(df2)[1])

	tables = c(df1, df2)

	df_total = merge(df1, df2, all=T)

	df_total$method = factor(df_total$m)

	print("hi")

	p1 = ggplot(data=df_total, aes(x=times, y=score, 
				group=method, color=method)) +
		    geom_point(size=0.7) + 
		    geom_line(size=0.3) +
    		stat_smooth(se = FALSE, size=1.5) +
		    ylim(c(0, 500)) + 
		    xlab("Second") +
		    ylab("Mean Score") +
		    ggtitle("Traditional DNN v.s. Q-learning") +
    ggsave(paste("./plots/dnnvsq_smooth.png", sep = ""), width=10,
    	   height=6, dpi=300)
}

# plot_single()
# plot_multiple()
# plot_alpha()
# plot_mini()
# plot_eta()
# plot_nn()
plot_e()
# plot_p()
# plot_msize()




