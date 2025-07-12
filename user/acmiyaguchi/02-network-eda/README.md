# network eda

This directory is for figuring out the network structure of the wikipedia dump from 2025-07-01.
We have the data in parquet, but this contains far more columns than we actually need to look at.

## Goals

Here is the goal state:

### **1. Initial Data Profiling**

- **Page Type Distribution**: Analyze the `page` dataset to determine the number of pages in each namespace (articles, redirects, talk pages, user pages, etc.).
- **Raw Edge Counts**: Calculate the total number of links in the raw `pagelinks` and `categorylinks` datasets before any filtering.

---

### **2. Pruning and Graph Structure**

- **Article-to-Article Edge Count**: Determine the number of edges that exist exclusively between main content articles after filtering out all other page types.
- **Degree Distribution**: On the pruned article-only graph, analyze the in-degree and out-degree distribution to understand the link structure (e.g., to confirm it follows a power law).
- **Redirect Analysis**: Investigate the redirect structure, including the total number of redirects and the longest redirect chains.

---

### **3. Artifact Generation**

- **Final Edge List**: Produce a clean Parquet file containing only the pruned, redirect-resolved edges between main articles.
