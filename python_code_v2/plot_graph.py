import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def parse_results(file):
    data =  {
        "balancedness": "",
        "distance": "",
        "points": []
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
    for i in range(2, len(lines) - 1, 8):
        weight = lines[i].split(" ")[1].replace("\n", "")
        acc = float(lines[i + 1].split(":")[1])
        time = float(lines[i + 4].split(":")[1])
        data["points"].append((weight, acc, time))
    f.close()
    return data

def plot_graph(dataList):
    colors = ["red", "green", "blue"]
    markers = ["o", "^"]
    dist = []
    for i in range(len(dataList)):
        color = colors[i]
        dist.append(dataList[i]["distance"])
        for j in range(len(dataList[i]["points"])):
            marker = ""
            if dataList[i]["points"][j][0] == "pnw":
                marker = markers[0]
            elif dataList[i]["points"][j][0] == "piwm":
                marker = markers[1]
            plt.scatter(dataList[i]["points"][j][1], dataList[i]["points"][j][2], color=color, marker=marker)
    #asume that all dataList elemnt have the same balancedness value
    plt.title(dataList[0]["balancedness"])
    plt.xlabel("Mean accuracy")
    plt.ylabel("Mean execution time")
    w = {"o": "pnw", "^": "piwm"}
    d = {"red": "en", "green": "ca", "blue": "br"}
    legend_elements = [
        mlines.Line2D([], [], color=c, marker=m, linestyle='None',
        markersize=10, label=f'{d[c]}, {w[m]}')
        for m in markers for c in colors
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.show()

data = [
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes/mean10_results_distr_10000_balanced.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_ca/mean10_results_distr_10000_balanced.txt"),
    parse_results("/home/ruben/FIC/GRADO/Q8/TFG/distr_edge_classifier/python_code_v2/results_3_nodes_br/mean10_results_distr_10000_balanced.txt")
]

plot_graph(data)