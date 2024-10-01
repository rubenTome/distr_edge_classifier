import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def parse_results(file):
    data =  {
        "balancedness": "",
        "distance": "",
        "points": [],
        "datasets": []
    }
    f = open(file, 'r')
    lines = f.readlines()
    data["balancedness"] = lines[1].split(" ")[5]
    if "_ca" in file:
        data["distance"] = "ca"
    elif "_br" in file:
        data["distance"] = "br"
    else:
        data["distance"] = "en"
    for i in range(1, len(lines) - 1, 8):
        dataset = lines[i].split(" ")[6]
        weight = lines[i + 1].split(" ")[1].replace("\n", "")
        acc = float(lines[i + 2].split(":")[1])
        time = float(lines[i + 5].split(":")[1])
        data["points"].append((weight, acc, time))
        data["datasets"].append(dataset)
    f.close()
    return data

def plot_graph(dataList, selectedDs):
    colors = ["red", "green", "blue"]
    marker = "o"
    selectedW = "piwm"
    dist = []
    w = {"piwm": "PIW"}
    for i in range(len(dataList)):
        color = colors[i]
        dist.append(dataList[i]["distance"])
        for j in range(len(dataList[i]["points"])):
            if (dataList[i]["points"][j][0] == selectedW 
                and selectedDs in dataList[i]["datasets"][j]):
                plt.scatter(dataList[i]["points"][j][1], dataList[i]["points"][j][2], s=60, color=color, marker=marker)
    #asume that all dataList elemnt have the same balancedness value
    if dataList[0]["balancedness"] == "perturbated":
        plt.title(selectedDs + " dataset, " 
                  + "unbalanced scenario, " 
                  + w[selectedW])
    else:
        plt.title(selectedDs + " dataset, " 
                  + dataList[0]["balancedness"] + " scenario, " 
                  + w[selectedW])
    plt.xlabel("Mean accuracy")
    plt.ylabel("Mean execution time")
    d = {"red": "Energy", "green": "Canberra", "blue": "Bray-Curtis"}
    legend_elements = [
        mlines.Line2D([], [], color=c, marker=marker, linestyle='None',
        markersize=10, label=f'{d[c]}')
        for c in colors
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.show()

data = [
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes/mean10_results_distr_10000_balanced.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_ca/mean10_results_distr_10000_balanced.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_br/mean10_results_distr_10000_balanced.txt")
]
for d in ["covtype", "HIGGS", "connect-4", "mnist"]:
    plot_graph(data, d)

data = [
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes/mean10_results_distr_10000_perturbated.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_ca/mean10_results_distr_10000_perturbated.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_br/mean10_results_distr_10000_perturbated.txt")
]
for d in ["covtype", "HIGGS", "connect-4", "mnist"]:
    plot_graph(data, d)