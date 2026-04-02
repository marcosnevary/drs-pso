dimensions = [30, 500]
benchmarks = df["benchmark"].unique()[:8]

# Aumentamos o figsize vertical para 12 para dar respiro à legenda
fig, axes = plt.subplots(4, 4, figsize=(15, 10))

handles, labels = [], []

for i, benchmark in enumerate(benchmarks):
    row = i // 2
    col_pair = (i % 2) * 2

    for j, dim in enumerate(dimensions):
        ax = axes[row, col_pair + j]
        subset = df[(df["dimension"] == dim) & (df["benchmark"] == benchmark)]

        if not subset.empty:
            plot_data = []
            for algorithm in subset["algorithm"].unique():
                alg_subset = subset[subset["algorithm"] == algorithm]
                df_fitness = pd.DataFrame(alg_subset["fitness_history"].tolist())
                df_long = df_fitness.melt(var_name="Iteration", value_name="Fitness")
                df_long["Algorithm"] = algorithm
                plot_data.append(df_long)

            df_plot = pd.concat(plot_data)

            sns.lineplot(
                data=df_plot,
                x="Iteration",
                y="Fitness",
                hue="Algorithm",
                ax=ax,
                errorbar="sd",
                legend=True,  # Ativamos temporariamente para capturar os handles
            )

            # Se ainda não pegamos a legenda, pegamos agora e removemos do eixo
            if not handles:
                handles, labels = ax.get_legend_handles_labels()

            ax.get_legend().remove()  # Remove a legenda local imediatamente

        ax.set_title(f"{benchmark} (d = {dim})", fontsize=10)
        ax.set_xlabel("Iteration" if row == 3 else "")
        ax.set_ylabel("Fitness" if col_pair + j == 0 else "")
        ax.grid(visible=True, linestyle="--", alpha=0.25)
        sns.despine(ax=ax)

# Criar a legenda global
if handles:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.98),  # Ajuste fino da posição vertical
        frameon=False,
        fontsize=12,
    )

# Ajuste manual das margens para evitar conflito com a legenda
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.4, wspace=0.3)

plt.savefig("../results/convergence_plots.pdf", bbox_inches="tight")
plt.show()
