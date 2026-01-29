'''
Centralise management of saving results
'''

# save results to /results for later use and plotting
def save_results(results, name):
    path = f"results/{name}.csv"
    results.to_csv(path, index=False)
    return path