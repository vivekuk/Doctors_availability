# Visualization example: Scatter plot
plt.scatter(df['Speciality'], df['kmeans_Cluster'])
plt.title('Scatter Plot of Speciality vs kmeans_Cluster')
plt.xlabel('Speciality')
plt.ylabel('kmeans_Cluster')
plt.show()
