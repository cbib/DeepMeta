import math

import numpy as np
from tqdm import tqdm


def count_pixels(mask_list):
    count0 = 0
    count1_pred = 0
    for j in tqdm(range(0, len(mask_list))):
        count1_pred += np.count_nonzero(mask_list[j])
        count0 += 16384 - np.count_nonzero(mask_list[j])
    return count0, count1_pred


def process_intersections(mask_pred_list, mask_gt_list):
    intersection0_tab = []
    intersection1_tab = []

    erreur_pred0_tab = []
    erreur_pred1_tab = []
    for i, mask in tqdm(enumerate(mask_pred_list)):
        mask_gt = mask_gt_list[i]
        intersection0 = 0
        intersection1 = 0
        erreur_pred0 = 0
        erreur_pred1 = 0
        for lx in range(0, len(mask)):
            for m in range(0, len(mask[0])):
                if mask[lx][m] == 0 and mask_gt[lx][m] == 0:
                    intersection0 += 1
                elif mask[lx][m] == 0 and mask_gt[lx][m] == 1:
                    erreur_pred1 += 1
                elif mask[lx][m] == 1 and mask_gt[lx][m] == 0:
                    erreur_pred0 += 1
                elif mask[lx][m] == 1 and mask_gt[lx][m] == 1:
                    intersection1 += 1
                else:
                    print(
                        "something weird happened : "
                        + str(mask_pred_list[lx][m])
                        + " : "
                        + str(mask_gt_list[lx][m])
                        + "\n"
                    )

        intersection0_tab.append(intersection0)
        intersection1_tab.append(intersection1)
        erreur_pred0_tab.append(erreur_pred0)
        erreur_pred1_tab.append(erreur_pred1)
    return intersection0_tab, intersection1_tab, erreur_pred0_tab, erreur_pred1_tab


def process_mcc(
    intersection0_tab, intersection1_tab, erreur_pred1_tab, erreur_pred0_tab
):
    TP = sum(intersection1_tab)
    TN = sum(intersection0_tab)
    FP = sum(erreur_pred1_tab)
    FN = sum(erreur_pred0_tab)

    numerateur = TP * TN - FP * FN
    denominateur = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    return numerateur / math.sqrt(denominateur)


def do_stats(mask_pred_list, mask_gt_list, save_path):
    f = open(save_path + "finestat.txt", "w")
    f.write("STAT PREDICTION" + "\n")

    count0, count1_pred = count_pixels(mask_pred_list)
    f.write("PRED : \n")
    f.write("pixel count0 : " + str(count0) + "\n")
    f.write("pixel count1 : " + str(count1_pred) + "\n")

    # GT
    count0, count1_GT = count_pixels(mask_gt_list)
    f.write("GT : \n")
    f.write("pixel count0 : " + str(count0) + "\n")
    f.write("pixel count1 : " + str(count1_GT) + "\n")

    (
        intersection0_tab,
        intersection1_tab,
        erreur_pred0_tab,
        erreur_pred1_tab,
    ) = process_intersections(mask_pred_list, mask_gt_list)

    f.write("Moyenne de pixels blanc bien classifiés (moyenne sur toutes les images)\n")
    moyenne1 = sum(intersection1_tab) / len(mask_pred_list)
    f.write(str(moyenne1) + "\n")

    f.write("Moyenne de pixels blanc images GT (moyenne sur toutes les images)\n")
    moyenne2 = count1_GT / len(mask_pred_list)
    f.write(str(moyenne2) + "\n")

    f.write("Moyenne de pixels blanc images predites (moyenne sur toutes les images)\n")
    moyenne3 = count1_pred / len(mask_pred_list)
    f.write(str(moyenne3) + "\n")

    f.write("Pourcentage de pixels blanc bien classifiés\n")
    # c'est intersection1 sur la somme des pixels blanc (intersection1 + erreur_pred1)
    pourcentage = sum(intersection1_tab) / (
        sum(intersection1_tab) + sum(erreur_pred1_tab)
    )
    f.write(str(pourcentage) + "\n")

    mcc = process_mcc(
        intersection0_tab, intersection1_tab, erreur_pred1_tab, erreur_pred0_tab
    )

    f.write("MCC : \n")
    f.write(str(mcc))
    f.close()
