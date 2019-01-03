plt.figure(figsize = (6,8))

# gs.update(left = 0.05, right = 1, bottom = 0.1, top = 0.98, wspace = 0.1, hspace = 0.05)

fig,ax=plt.subplots()

# wild plotting

ax = plt.gca()

data = [[6, 78, "Zhang et al, 2017"],
		[10, 80, "paper 2"]]

ax.scatter(6, 78,marker="x")
ax.annotate("Zhang et al, 2017", (5, 79.8))

ax.scatter(3, 74, marker="x")
ax.annotate("Vasileios et al, 2017", (1, 75.5))

ax.scatter(6, 57,marker="x")
ax.annotate("Abdelkareem et al, 2015", (4, 59))

ax.scatter(1, 71.5,marker="x")
ax.annotate("Edison et al, 2015", (0, 68))

ax.set_xlim(0, 10)

ax.set_xlabel('In-wild, by hours')
ax.set_ylabel('f-score')
ax.set_ylim(40, 100)

plt.show()