# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from torch.cuda.amp import GradScaler, autocast
import pycocotools.mask as mask_util
import numpy as np
from datasets.coco_eval import CocoEvaluator, convert_to_xywh

freq_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202]
common_classes = [1, 5, 6, 7, 8, 9, 10, 16, 18, 22, 24, 25, 26, 28, 33, 37, 44, 46, 47, 54, 55, 62, 64, 67, 70, 72, 73, 74, 84, 91, 92, 97, 98, 100, 101, 102, 103, 107, 108, 111, 120, 121, 122, 124, 128, 129, 134, 135, 141, 145, 148, 149, 153, 156, 157, 158, 162, 163, 165, 166, 168, 170, 174, 176, 180, 184, 186, 187, 188, 190, 191, 193, 197, 198, 199, 200, 201, 205, 206, 211, 212, 213, 216, 219, 220, 221, 224, 227, 228, 239, 241, 242, 243, 246, 248, 249, 256, 260, 263, 264, 267, 268, 273, 274, 278, 279, 280, 283, 286, 288, 289, 290, 293, 308, 311, 312, 314, 315, 318, 320, 322, 325, 327, 328, 329, 332, 334, 335, 336, 337, 339, 340, 341, 343, 345, 356, 359, 360, 363, 370, 371, 383, 384, 386, 391, 393, 396, 399, 402, 403, 406, 408, 412, 417, 418, 419, 423, 424, 425, 433, 434, 436, 438, 443, 448, 450, 453, 454, 455, 457, 460, 462, 463, 464, 465, 466, 468, 470, 471, 472, 473, 475, 476, 483, 484, 485, 487, 489, 490, 493, 494, 495, 497, 499, 501, 504, 505, 507, 511, 512, 519, 520, 522, 523, 525, 526, 529, 530, 531, 533, 537, 539, 546, 550, 552, 553, 554, 555, 558, 562, 564, 573, 576, 579, 581, 584, 587, 588, 590, 593, 596, 598, 600, 601, 604, 607, 608, 612, 613, 622, 623, 629, 636, 637, 649, 650, 652, 654, 656, 660, 666, 667, 673, 677, 680, 681, 682, 683, 684, 695, 696, 697, 699, 707, 711, 717, 718, 720, 721, 723, 725, 731, 732, 736, 737, 740, 741, 742, 744, 746, 747, 750, 753, 760, 761, 762, 763, 765, 767, 768, 770, 773, 774, 775, 777, 780, 786, 790, 791, 794, 795, 797, 801, 802, 807, 813, 819, 821, 825, 826, 830, 833, 834, 839, 840, 841, 842, 843, 844, 846, 847, 854, 857, 861, 863, 866, 867, 868, 870, 871, 872, 874, 875, 877, 878, 879, 882, 884, 888, 889, 893, 895, 897, 901, 906, 907, 909, 922, 926, 928, 929, 930, 932, 933, 934, 935, 936, 940, 946, 950, 954, 960, 963, 970, 971, 973, 977, 978, 984, 988, 989, 996, 997, 999, 1001, 1002, 1004, 1006, 1007, 1009, 1013, 1014, 1022, 1034, 1036, 1038, 1039, 1040, 1041, 1044, 1046, 1051, 1062, 1063, 1065, 1066, 1067, 1068, 1069, 1073, 1076, 1081, 1082, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1094, 1101, 1106, 1107, 1111, 1113, 1120, 1125, 1127, 1128, 1130, 1131, 1132, 1137, 1138, 1140, 1143, 1147, 1149, 1151, 1152, 1153, 1154, 1160, 1163, 1164, 1166, 1168, 1169, 1170, 1171, 1174, 1175, 1176, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1187, 1189, 1192, 1194, 1195, 1196, 1199, 1200, 1201, 1203]
rare_classes = [13, 14, 17, 20, 21, 30, 31, 38, 39, 40, 42, 49, 51, 52, 63, 69, 71, 78, 82, 85, 93, 105, 106, 113, 117, 119, 123, 126, 130, 131, 136, 140, 142, 144, 147, 151, 155, 159, 161, 164, 167, 172, 179, 182, 196, 202, 209, 210, 214, 215, 222, 223, 231, 233, 234, 236, 237, 238, 240, 244, 245, 247, 250, 251, 257, 258, 262, 265, 266, 269, 270, 275, 281, 282, 287, 291, 292, 294, 295, 300, 301, 302, 304, 307, 310, 313, 316, 317, 321, 323, 326, 331, 333, 348, 349, 352, 353, 354, 355, 357, 362, 364, 365, 366, 368, 374, 376, 381, 382, 388, 389, 397, 398, 400, 405, 407, 410, 413, 414, 416, 420, 426, 427, 428, 431, 432, 435, 439, 446, 449, 456, 458, 467, 478, 479, 480, 481, 482, 486, 488, 491, 492, 503, 506, 508, 509, 513, 516, 518, 527, 532, 535, 538, 541, 542, 543, 545, 551, 557, 560, 561, 567, 568, 571, 572, 574, 575, 577, 580, 582, 583, 585, 594, 597, 599, 602, 603, 606, 610, 616, 618, 619, 620, 625, 632, 634, 635, 638, 640, 646, 648, 651, 657, 662, 663, 664, 665, 671, 672, 674, 678, 686, 688, 690, 691, 693, 702, 710, 712, 714, 722, 727, 729, 730, 733, 743, 752, 754, 755, 758, 759, 764, 769, 772, 778, 779, 783, 784, 785, 787, 788, 792, 796, 803, 805, 808, 809, 810, 812, 815, 820, 822, 823, 824, 829, 831, 849, 850, 851, 852, 853, 855, 856, 858, 859, 862, 864, 869, 873, 883, 886, 887, 890, 891, 892, 894, 902, 905, 908, 913, 914, 917, 918, 920, 925, 931, 937, 938, 939, 941, 942, 944, 945, 952, 956, 958, 969, 972, 974, 975, 983, 985, 987, 990, 991, 992, 994, 998, 1003, 1005, 1010, 1012, 1015, 1016, 1028, 1029, 1030, 1031, 1032, 1047, 1048, 1049, 1053, 1054, 1057, 1058, 1075, 1080, 1084, 1116, 1118, 1119, 1124, 1126, 1129, 1135, 1144, 1145, 1146, 1148, 1150, 1157, 1158, 1159, 1165, 1167, 1193]
freq_common_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202, 1, 5, 6, 7, 8, 9, 10, 16, 18, 22, 24, 25, 26, 28, 33, 37, 44, 46, 47, 54, 55, 62, 64, 67, 70, 72, 73, 74, 84, 91, 92, 97, 98, 100, 101, 102, 103, 107, 108, 111, 120, 121, 122, 124, 128, 129, 134, 135, 141, 145, 148, 149, 153, 156, 157, 158, 162, 163, 165, 166, 168, 170, 174, 176, 180, 184, 186, 187, 188, 190, 191, 193, 197, 198, 199, 200, 201, 205, 206, 211, 212, 213, 216, 219, 220, 221, 224, 227, 228, 239, 241, 242, 243, 246, 248, 249, 256, 260, 263, 264, 267, 268, 273, 274, 278, 279, 280, 283, 286, 288, 289, 290, 293, 308, 311, 312, 314, 315, 318, 320, 322, 325, 327, 328, 329, 332, 334, 335, 336, 337, 339, 340, 341, 343, 345, 356, 359, 360, 363, 370, 371, 383, 384, 386, 391, 393, 396, 399, 402, 403, 406, 408, 412, 417, 418, 419, 423, 424, 425, 433, 434, 436, 438, 443, 448, 450, 453, 454, 455, 457, 460, 462, 463, 464, 465, 466, 468, 470, 471, 472, 473, 475, 476, 483, 484, 485, 487, 489, 490, 493, 494, 495, 497, 499, 501, 504, 505, 507, 511, 512, 519, 520, 522, 523, 525, 526, 529, 530, 531, 533, 537, 539, 546, 550, 552, 553, 554, 555, 558, 562, 564, 573, 576, 579, 581, 584, 587, 588, 590, 593, 596, 598, 600, 601, 604, 607, 608, 612, 613, 622, 623, 629, 636, 637, 649, 650, 652, 654, 656, 660, 666, 667, 673, 677, 680, 681, 682, 683, 684, 695, 696, 697, 699, 707, 711, 717, 718, 720, 721, 723, 725, 731, 732, 736, 737, 740, 741, 742, 744, 746, 747, 750, 753, 760, 761, 762, 763, 765, 767, 768, 770, 773, 774, 775, 777, 780, 786, 790, 791, 794, 795, 797, 801, 802, 807, 813, 819, 821, 825, 826, 830, 833, 834, 839, 840, 841, 842, 843, 844, 846, 847, 854, 857, 861, 863, 866, 867, 868, 870, 871, 872, 874, 875, 877, 878, 879, 882, 884, 888, 889, 893, 895, 897, 901, 906, 907, 909, 922, 926, 928, 929, 930, 932, 933, 934, 935, 936, 940, 946, 950, 954, 960, 963, 970, 971, 973, 977, 978, 984, 988, 989, 996, 997, 999, 1001, 1002, 1004, 1006, 1007, 1009, 1013, 1014, 1022, 1034, 1036, 1038, 1039, 1040, 1041, 1044, 1046, 1051, 1062, 1063, 1065, 1066, 1067, 1068, 1069, 1073, 1076, 1081, 1082, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1094, 1101, 1106, 1107, 1111, 1113, 1120, 1125, 1127, 1128, 1130, 1131, 1132, 1137, 1138, 1140, 1143, 1147, 1149, 1151, 1152, 1153, 1154, 1160, 1163, 1164, 1166, 1168, 1169, 1170, 1171, 1174, 1175, 1176, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1187, 1189, 1192, 1194, 1195, 1196, 1199, 1200, 1201, 1203]
freq_rare_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202, 13, 14, 17, 20, 21, 30, 31, 38, 39, 40, 42, 49, 51, 52, 63, 69, 71, 78, 82, 85, 93, 105, 106, 113, 117, 119, 123, 126, 130, 131, 136, 140, 142, 144, 147, 151, 155, 159, 161, 164, 167, 172, 179, 182, 196, 202, 209, 210, 214, 215, 222, 223, 231, 233, 234, 236, 237, 238, 240, 244, 245, 247, 250, 251, 257, 258, 262, 265, 266, 269, 270, 275, 281, 282, 287, 291, 292, 294, 295, 300, 301, 302, 304, 307, 310, 313, 316, 317, 321, 323, 326, 331, 333, 348, 349, 352, 353, 354, 355, 357, 362, 364, 365, 366, 368, 374, 376, 381, 382, 388, 389, 397, 398, 400, 405, 407, 410, 413, 414, 416, 420, 426, 427, 428, 431, 432, 435, 439, 446, 449, 456, 458, 467, 478, 479, 480, 481, 482, 486, 488, 491, 492, 503, 506, 508, 509, 513, 516, 518, 527, 532, 535, 538, 541, 542, 543, 545, 551, 557, 560, 561, 567, 568, 571, 572, 574, 575, 577, 580, 582, 583, 585, 594, 597, 599, 602, 603, 606, 610, 616, 618, 619, 620, 625, 632, 634, 635, 638, 640, 646, 648, 651, 657, 662, 663, 664, 665, 671, 672, 674, 678, 686, 688, 690, 691, 693, 702, 710, 712, 714, 722, 727, 729, 730, 733, 743, 752, 754, 755, 758, 759, 764, 769, 772, 778, 779, 783, 784, 785, 787, 788, 792, 796, 803, 805, 808, 809, 810, 812, 815, 820, 822, 823, 824, 829, 831, 849, 850, 851, 852, 853, 855, 856, 858, 859, 862, 864, 869, 873, 883, 886, 887, 890, 891, 892, 894, 902, 905, 908, 913, 914, 917, 918, 920, 925, 931, 937, 938, 939, 941, 942, 944, 945, 952, 956, 958, 969, 972, 974, 975, 983, 985, 987, 990, 991, 992, 994, 998, 1003, 1005, 1010, 1012, 1015, 1016, 1028, 1029, 1030, 1031, 1032, 1047, 1048, 1049, 1053, 1054, 1057, 1058, 1075, 1080, 1084, 1116, 1118, 1119, 1124, 1126, 1129, 1135, 1144, 1145, 1146, 1148, 1150, 1157, 1158, 1159, 1165, 1167, 1193]

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def lvis_evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, label_map, num_classes):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    lvis_results = []

    cat2label = data_loader.dataset.cat2label
    label2cat = {v: k for k, v in cat2label.items()}

    # sampling
    class_scores = dict.fromkeys(range(1, num_classes), [])
    ids = dict.fromkeys(range(1, num_classes), [])
    image_ids = dict.fromkeys(range(1, num_classes), [])

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        scores, target_ids, target_classes, target_image_ids = criterion(outputs, targets)

        for i in range(len(targets)):
            for j in range(target_classes.size(0)):
                for (k1, v1), (k2, v2), (k3, v3) in zip(class_scores.items(), ids.items(), image_ids.items()):
                    v1_copy = v1.copy()
                    v2_copy = v2.copy()
                    v3_copy = v3.copy()
                    if target_classes[j].int() == k1:
                        class_scores[k1].append(scores[j])
                        ids[k1].append(target_ids[j])
                        image_ids[k1].append(target_image_ids[j])
                    else:
                        class_scores[k1] = v1_copy
                        ids[k1] = v2_copy
                        image_ids[k1] = v3_copy
    #     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #     results, topk_boxes = postprocessors["bbox"](outputs, orig_target_sizes)
    #     if "segm" in postprocessors.keys():
    #         target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #         outputs_masks = outputs["pred_masks"].squeeze(2)
    #
    #         bs = len(topk_boxes)
    #         outputs_masks_new = [[] for _ in range(bs)]
    #         for b in range(bs):
    #             for index in topk_boxes[b]:
    #                 outputs_masks_new[b].append(outputs_masks[b : b + 1, index : index + 1, :, :])
    #         for b in range(bs):
    #             outputs_masks_new[b] = torch.cat(outputs_masks_new[b], 1)
    #         outputs["pred_masks"] = torch.cat(outputs_masks_new, 0)
    #
    #         results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
    #
    #     for target, output in zip(targets, results):
    #         image_id = target["image_id"].item()
    #
    #         if "masks" in output.keys():
    #             masks = output["masks"].data.cpu().numpy()
    #             masks = masks > 0.5
    #             rles = [
    #                 mask_util.encode(
    #                     np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
    #                 )[0]
    #                 for mask in masks
    #             ]
    #             for rle in rles:
    #                 rle["counts"] = rle["counts"].decode("utf-8")
    #
    #         boxes = convert_to_xywh(output["boxes"])
    #         for ind in range(len(output["scores"])):
    #             temp = {
    #                 "image_id": image_id,
    #                 "score": output["scores"][ind].item(),
    #                 "category_id": output["labels"][ind].item(),
    #                 "bbox": boxes[ind].tolist(),
    #             }
    #             if label_map:
    #                 temp["category_id"] = label2cat[temp["category_id"]]
    #             if "masks" in output.keys():
    #                 temp["segmentation"] = rles[ind]
    #
    #             lvis_results.append(temp)
    #
    # # rank = torch.distributed.get_rank()
    # # torch.save(lvis_results, output_dir + f"/pred_{rank}.pth")
    # #
    # #
    # # # gather the stats from all processes
    # # metric_logger.synchronize_between_processes()
    # # torch.distributed.barrier()
    # # if rank == 0:
    # #     world_size = torch.distributed.get_world_size()
    # #     for i in range(1, world_size):
    # #         temp = torch.load(output_dir + f"/pred_{i}.pth")
    # #         lvis_results += temp
    # #
    # #     from lvis import LVISEval, LVISResults
    # #
    # #     lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
    # #     for iou_type in iou_types:
    # #         lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
    # #         lvis_eval.run()
    # #         lvis_eval.print_results()
    # # torch.distributed.barrier()
    # # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # # return stats, None
    #
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    #
    # from lvis import LVISEval, LVISResults
    #
    # lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
    # for iou_type in iou_types:
    #     lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
    #     lvis_eval.run()
    #     lvis_eval.print_results()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return stats, None
    return None, None, class_scores, ids, image_ids