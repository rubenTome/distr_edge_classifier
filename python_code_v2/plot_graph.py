import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def parse_results(file):
    data =  {
        "balancedness": "",
        "distance": "",
        "points": [],
        "datasets": [],
        "classifiers": []
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
        cw = lines[i + 1].split(" ")
        classifier = cw[0]
        weight = cw[1].replace("\n", "")
        acc = float(lines[i + 2].split(":")[1])
        time = float(lines[i + 5].split(":")[1])
        data["points"].append((weight, acc, time))
        data["classifiers"].append(classifier)
        data["datasets"].append(dataset)
    f.close()
    return data

def plot_graph(dataList, selectedDs):
    colors = ["red", "green", "blue"]
    markers = ["o", "^", "P", "*"]
    selectedW = "piwm"
    selectedClass = ["knn", "rf", "svm", "xgb"]
    dist = []
    w = {"piwm": "PIW"}
    for i in range(len(dataList)):
        color = colors[i]
        dist.append(dataList[i]["distance"])
        for j in range(len(dataList[i]["points"])):
            if (dataList[i]["points"][j][0] == selectedW 
                and selectedDs in dataList[i]["datasets"][j]):
                marker = ""
                if dataList[i]["classifiers"][j] == "knn":
                    marker = markers[0]
                elif dataList[i]["classifiers"][j] == "rf":
                    marker = markers[1]
                elif dataList[i]["classifiers"][j] == "svm":
                    marker = markers[2]
                else:
                    marker = markers[3]
                plt.scatter(dataList[i]["points"][j][1], dataList[i]["points"][j][2], s=150, color=color, marker=marker)
    #asume that all dataList elemnt have the same balancedness value
    if dataList[0]["balancedness"] == "perturbated":
        plt.title(selectedDs.upper() + " DATASET, " 
                  + "UNBALANCED SCENARIO, " 
                  + w[selectedW])
    else:
        plt.title(selectedDs.upper() + " DATASET, " 
                  + dataList[0]["balancedness"].upper() + " SCENARIO, " 
                  + w[selectedW], fontsize="15")
    plt.xlabel("MEAN ACCURACY", fontsize="15")
    plt.ylabel("MEAN EXECUTION TIME", fontsize="15")
    d = {"red": "Energy", "green": "Canberra", "blue": "Bray-Curtis"}
    m = {selectedClass[0]: markers[0],
         selectedClass[1]: markers[1], 
         selectedClass[2]: markers[2], 
         selectedClass[3]: markers[3]}
    legend_elements = [
        mlines.Line2D([], [], color=c, marker=m[cl], linestyle='None',
        markersize=10, label=f'{d[c].upper()}, {cl.upper()}')
        for c in colors for cl in selectedClass
    ]
    #plt.legend(handles=legend_elements, ncol=3, loc='upper left', fontsize="15")
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