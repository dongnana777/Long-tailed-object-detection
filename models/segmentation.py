# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

freq_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202]
common_classes = [1, 5, 6, 7, 8, 9, 10, 16, 18, 22, 24, 25, 26, 28, 33, 37, 44, 46, 47, 54, 55, 62, 64, 67, 70, 72, 73, 74, 84, 91, 92, 97, 98, 100, 101, 102, 103, 107, 108, 111, 120, 121, 122, 124, 128, 129, 134, 135, 141, 145, 148, 149, 153, 156, 157, 158, 162, 163, 165, 166, 168, 170, 174, 176, 180, 184, 186, 187, 188, 190, 191, 193, 197, 198, 199, 200, 201, 205, 206, 211, 212, 213, 216, 219, 220, 221, 224, 227, 228, 239, 241, 242, 243, 246, 248, 249, 256, 260, 263, 264, 267, 268, 273, 274, 278, 279, 280, 283, 286, 288, 289, 290, 293, 308, 311, 312, 314, 315, 318, 320, 322, 325, 327, 328, 329, 332, 334, 335, 336, 337, 339, 340, 341, 343, 345, 356, 359, 360, 363, 370, 371, 383, 384, 386, 391, 393, 396, 399, 402, 403, 406, 408, 412, 417, 418, 419, 423, 424, 425, 433, 434, 436, 438, 443, 448, 450, 453, 454, 455, 457, 460, 462, 463, 464, 465, 466, 468, 470, 471, 472, 473, 475, 476, 483, 484, 485, 487, 489, 490, 493, 494, 495, 497, 499, 501, 504, 505, 507, 511, 512, 519, 520, 522, 523, 525, 526, 529, 530, 531, 533, 537, 539, 546, 550, 552, 553, 554, 555, 558, 562, 564, 573, 576, 579, 581, 584, 587, 588, 590, 593, 596, 598, 600, 601, 604, 607, 608, 612, 613, 622, 623, 629, 636, 637, 649, 650, 652, 654, 656, 660, 666, 667, 673, 677, 680, 681, 682, 683, 684, 695, 696, 697, 699, 707, 711, 717, 718, 720, 721, 723, 725, 731, 732, 736, 737, 740, 741, 742, 744, 746, 747, 750, 753, 760, 761, 762, 763, 765, 767, 768, 770, 773, 774, 775, 777, 780, 786, 790, 791, 794, 795, 797, 801, 802, 807, 813, 819, 821, 825, 826, 830, 833, 834, 839, 840, 841, 842, 843, 844, 846, 847, 854, 857, 861, 863, 866, 867, 868, 870, 871, 872, 874, 875, 877, 878, 879, 882, 884, 888, 889, 893, 895, 897, 901, 906, 907, 909, 922, 926, 928, 929, 930, 932, 933, 934, 935, 936, 940, 946, 950, 954, 960, 963, 970, 971, 973, 977, 978, 984, 988, 989, 996, 997, 999, 1001, 1002, 1004, 1006, 1007, 1009, 1013, 1014, 1022, 1034, 1036, 1038, 1039, 1040, 1041, 1044, 1046, 1051, 1062, 1063, 1065, 1066, 1067, 1068, 1069, 1073, 1076, 1081, 1082, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1094, 1101, 1106, 1107, 1111, 1113, 1120, 1125, 1127, 1128, 1130, 1131, 1132, 1137, 1138, 1140, 1143, 1147, 1149, 1151, 1152, 1153, 1154, 1160, 1163, 1164, 1166, 1168, 1169, 1170, 1171, 1174, 1175, 1176, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1187, 1189, 1192, 1194, 1195, 1196, 1199, 1200, 1201, 1203]
rare_classes = [13, 14, 17, 20, 21, 30, 31, 38, 39, 40, 42, 49, 51, 52, 63, 69, 71, 78, 82, 85, 93, 105, 106, 113, 117, 119, 123, 126, 130, 131, 136, 140, 142, 144, 147, 151, 155, 159, 161, 164, 167, 172, 179, 182, 196, 202, 209, 210, 214, 215, 222, 223, 231, 233, 234, 236, 237, 238, 240, 244, 245, 247, 250, 251, 257, 258, 262, 265, 266, 269, 270, 275, 281, 282, 287, 291, 292, 294, 295, 300, 301, 302, 304, 307, 310, 313, 316, 317, 321, 323, 326, 331, 333, 348, 349, 352, 353, 354, 355, 357, 362, 364, 365, 366, 368, 374, 376, 381, 382, 388, 389, 397, 398, 400, 405, 407, 410, 413, 414, 416, 420, 426, 427, 428, 431, 432, 435, 439, 446, 449, 456, 458, 467, 478, 479, 480, 481, 482, 486, 488, 491, 492, 503, 506, 508, 509, 513, 516, 518, 527, 532, 535, 538, 541, 542, 543, 545, 551, 557, 560, 561, 567, 568, 571, 572, 574, 575, 577, 580, 582, 583, 585, 594, 597, 599, 602, 603, 606, 610, 616, 618, 619, 620, 625, 632, 634, 635, 638, 640, 646, 648, 651, 657, 662, 663, 664, 665, 671, 672, 674, 678, 686, 688, 690, 691, 693, 702, 710, 712, 714, 722, 727, 729, 730, 733, 743, 752, 754, 755, 758, 759, 764, 769, 772, 778, 779, 783, 784, 785, 787, 788, 792, 796, 803, 805, 808, 809, 810, 812, 815, 820, 822, 823, 824, 829, 831, 849, 850, 851, 852, 853, 855, 856, 858, 859, 862, 864, 869, 873, 883, 886, 887, 890, 891, 892, 894, 902, 905, 908, 913, 914, 917, 918, 920, 925, 931, 937, 938, 939, 941, 942, 944, 945, 952, 956, 958, 969, 972, 974, 975, 983, 985, 987, 990, 991, 992, 994, 998, 1003, 1005, 1010, 1012, 1015, 1016, 1028, 1029, 1030, 1031, 1032, 1047, 1048, 1049, 1053, 1054, 1057, 1058, 1075, 1080, 1084, 1116, 1118, 1119, 1124, 1126, 1129, 1135, 1144, 1145, 1146, 1148, 1150, 1157, 1158, 1159, 1165, 1167, 1193]
freq_common_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202, 1, 5, 6, 7, 8, 9, 10, 16, 18, 22, 24, 25, 26, 28, 33, 37, 44, 46, 47, 54, 55, 62, 64, 67, 70, 72, 73, 74, 84, 91, 92, 97, 98, 100, 101, 102, 103, 107, 108, 111, 120, 121, 122, 124, 128, 129, 134, 135, 141, 145, 148, 149, 153, 156, 157, 158, 162, 163, 165, 166, 168, 170, 174, 176, 180, 184, 186, 187, 188, 190, 191, 193, 197, 198, 199, 200, 201, 205, 206, 211, 212, 213, 216, 219, 220, 221, 224, 227, 228, 239, 241, 242, 243, 246, 248, 249, 256, 260, 263, 264, 267, 268, 273, 274, 278, 279, 280, 283, 286, 288, 289, 290, 293, 308, 311, 312, 314, 315, 318, 320, 322, 325, 327, 328, 329, 332, 334, 335, 336, 337, 339, 340, 341, 343, 345, 356, 359, 360, 363, 370, 371, 383, 384, 386, 391, 393, 396, 399, 402, 403, 406, 408, 412, 417, 418, 419, 423, 424, 425, 433, 434, 436, 438, 443, 448, 450, 453, 454, 455, 457, 460, 462, 463, 464, 465, 466, 468, 470, 471, 472, 473, 475, 476, 483, 484, 485, 487, 489, 490, 493, 494, 495, 497, 499, 501, 504, 505, 507, 511, 512, 519, 520, 522, 523, 525, 526, 529, 530, 531, 533, 537, 539, 546, 550, 552, 553, 554, 555, 558, 562, 564, 573, 576, 579, 581, 584, 587, 588, 590, 593, 596, 598, 600, 601, 604, 607, 608, 612, 613, 622, 623, 629, 636, 637, 649, 650, 652, 654, 656, 660, 666, 667, 673, 677, 680, 681, 682, 683, 684, 695, 696, 697, 699, 707, 711, 717, 718, 720, 721, 723, 725, 731, 732, 736, 737, 740, 741, 742, 744, 746, 747, 750, 753, 760, 761, 762, 763, 765, 767, 768, 770, 773, 774, 775, 777, 780, 786, 790, 791, 794, 795, 797, 801, 802, 807, 813, 819, 821, 825, 826, 830, 833, 834, 839, 840, 841, 842, 843, 844, 846, 847, 854, 857, 861, 863, 866, 867, 868, 870, 871, 872, 874, 875, 877, 878, 879, 882, 884, 888, 889, 893, 895, 897, 901, 906, 907, 909, 922, 926, 928, 929, 930, 932, 933, 934, 935, 936, 940, 946, 950, 954, 960, 963, 970, 971, 973, 977, 978, 984, 988, 989, 996, 997, 999, 1001, 1002, 1004, 1006, 1007, 1009, 1013, 1014, 1022, 1034, 1036, 1038, 1039, 1040, 1041, 1044, 1046, 1051, 1062, 1063, 1065, 1066, 1067, 1068, 1069, 1073, 1076, 1081, 1082, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1094, 1101, 1106, 1107, 1111, 1113, 1120, 1125, 1127, 1128, 1130, 1131, 1132, 1137, 1138, 1140, 1143, 1147, 1149, 1151, 1152, 1153, 1154, 1160, 1163, 1164, 1166, 1168, 1169, 1170, 1171, 1174, 1175, 1176, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1187, 1189, 1192, 1194, 1195, 1196, 1199, 1200, 1201, 1203]
freq_rare_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202, 13, 14, 17, 20, 21, 30, 31, 38, 39, 40, 42, 49, 51, 52, 63, 69, 71, 78, 82, 85, 93, 105, 106, 113, 117, 119, 123, 126, 130, 131, 136, 140, 142, 144, 147, 151, 155, 159, 161, 164, 167, 172, 179, 182, 196, 202, 209, 210, 214, 215, 222, 223, 231, 233, 234, 236, 237, 238, 240, 244, 245, 247, 250, 251, 257, 258, 262, 265, 266, 269, 270, 275, 281, 282, 287, 291, 292, 294, 295, 300, 301, 302, 304, 307, 310, 313, 316, 317, 321, 323, 326, 331, 333, 348, 349, 352, 353, 354, 355, 357, 362, 364, 365, 366, 368, 374, 376, 381, 382, 388, 389, 397, 398, 400, 405, 407, 410, 413, 414, 416, 420, 426, 427, 428, 431, 432, 435, 439, 446, 449, 456, 458, 467, 478, 479, 480, 481, 482, 486, 488, 491, 492, 503, 506, 508, 509, 513, 516, 518, 527, 532, 535, 538, 541, 542, 543, 545, 551, 557, 560, 561, 567, 568, 571, 572, 574, 575, 577, 580, 582, 583, 585, 594, 597, 599, 602, 603, 606, 610, 616, 618, 619, 620, 625, 632, 634, 635, 638, 640, 646, 648, 651, 657, 662, 663, 664, 665, 671, 672, 674, 678, 686, 688, 690, 691, 693, 702, 710, 712, 714, 722, 727, 729, 730, 733, 743, 752, 754, 755, 758, 759, 764, 769, 772, 778, 779, 783, 784, 785, 787, 788, 792, 796, 803, 805, 808, 809, 810, 812, 815, 820, 822, 823, 824, 829, 831, 849, 850, 851, 852, 853, 855, 856, 858, 859, 862, 864, 869, 873, 883, 886, 887, 890, 891, 892, 894, 902, 905, 908, 913, 914, 917, 918, 920, 925, 931, 937, 938, 939, 941, 942, 944, 945, 952, 956, 958, 969, 972, 974, 975, 983, 985, 987, 990, 991, 992, 994, 998, 1003, 1005, 1010, 1012, 1015, 1016, 1028, 1029, 1030, 1031, 1032, 1047, 1048, 1049, 1053, 1054, 1057, 1058, 1075, 1080, 1084, 1116, 1118, 1119, 1124, 1126, 1129, 1135, 1144, 1145, 1146, 1148, 1150, 1157, 1158, 1159, 1165, 1167, 1193]

freq_common_30_100_classes = [2, 3, 4, 11, 12, 15, 19, 23, 27, 29, 32, 34, 35, 36, 41, 43, 45, 48, 50, 53, 56, 57, 58, 59, 60, 61, 65, 66, 68, 75, 76, 77, 79, 80, 81, 83, 86, 87, 88, 89, 90, 94, 95, 96, 99, 104, 109, 110, 112, 114, 115, 116, 118, 125, 127, 132, 133, 137, 138, 139, 143, 146, 150, 152, 154, 160, 169, 171, 173, 175, 177, 178, 181, 183, 185, 189, 192, 194, 195, 203, 204, 207, 208, 217, 218, 225, 226, 229, 230, 232, 235, 252, 253, 254, 255, 259, 261, 271, 272, 276, 277, 284, 285, 296, 297, 298, 299, 303, 305, 306, 309, 319, 324, 330, 338, 342, 344, 346, 347, 350, 351, 358, 361, 367, 369, 372, 373, 375, 377, 378, 379, 380, 385, 387, 390, 392, 394, 395, 401, 404, 409, 411, 415, 421, 422, 429, 430, 437, 440, 441, 442, 444, 445, 447, 451, 452, 459, 461, 469, 474, 477, 496, 498, 500, 502, 510, 514, 515, 517, 521, 524, 528, 534, 536, 540, 544, 547, 548, 549, 556, 559, 563, 565, 566, 569, 570, 578, 586, 589, 591, 592, 595, 605, 609, 611, 614, 615, 617, 621, 624, 626, 627, 628, 630, 631, 633, 639, 641, 642, 643, 644, 645, 647, 653, 655, 658, 659, 661, 668, 669, 670, 675, 676, 679, 685, 687, 689, 692, 694, 698, 700, 701, 703, 704, 705, 706, 708, 709, 713, 715, 716, 719, 724, 726, 728, 734, 735, 738, 739, 745, 748, 749, 751, 756, 757, 766, 771, 776, 781, 782, 789, 793, 798, 799, 800, 804, 806, 811, 814, 816, 817, 818, 827, 828, 832, 835, 836, 837, 838, 845, 848, 860, 865, 876, 880, 881, 885, 896, 898, 899, 900, 903, 904, 910, 911, 912, 915, 916, 919, 921, 923, 924, 927, 943, 947, 948, 949, 951, 953, 955, 957, 959, 961, 962, 964, 965, 966, 967, 968, 976, 979, 980, 981, 982, 986, 993, 995, 1000, 1008, 1011, 1017, 1018, 1019, 1020, 1021, 1023, 1024, 1025, 1026, 1027, 1033, 1035, 1037, 1042, 1043, 1045, 1050, 1052, 1055, 1056, 1059, 1060, 1061, 1064, 1070, 1071, 1072, 1074, 1077, 1078, 1079, 1083, 1091, 1093, 1095, 1096, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1108, 1109, 1110, 1112, 1114, 1115, 1117, 1121, 1122, 1123, 1133, 1134, 1136, 1139, 1141, 1142, 1155, 1156, 1161, 1162, 1172, 1173, 1177, 1178, 1186, 1188, 1190, 1191, 1197, 1198, 1202,1, 7, 16, 18, 22, 24, 25, 26, 28, 33, 37, 44, 47, 54, 62, 67, 70, 72, 84, 92, 97, 103, 107, 108, 111, 120, 122, 124, 128, 129, 141, 145, 148, 153, 156, 158, 162, 165, 166, 174, 176, 184, 186, 190, 191, 198, 199, 206, 212, 216, 220, 239, 241, 246, 249, 256, 260, 263, 267, 268, 273, 278, 288, 290, 293, 311, 312, 315, 322, 327, 328, 329, 334, 335, 337, 359, 363, 370, 386, 393, 406, 408, 412, 417, 418, 419, 424, 433, 436, 438, 450, 455, 457, 464, 473, 475, 484, 487, 493, 494, 499, 501, 507, 511, 512, 522, 523, 529, 537, 539, 546, 550, 552, 553, 554, 555, 558, 562, 579, 584, 593, 596, 601, 604, 612, 613, 622, 629, 636, 649, 667, 673, 677, 680, 681, 683, 695, 696, 697, 699, 707, 723, 725, 736, 737, 740, 741, 744, 746, 747, 753, 762, 767, 768, 773, 774, 786, 790, 794, 801, 802, 819, 833, 839, 841, 843, 847, 854, 861, 863, 866, 867, 870, 871, 872, 877, 888, 889, 893, 901, 906, 922, 926, 928, 932, 935, 950, 954, 970, 973, 977, 996, 999, 1002, 1006, 1009, 1013, 1014, 1022, 1034, 1036, 1040, 1041, 1046, 1051, 1065, 1066, 1068, 1069, 1073, 1076, 1082, 1085, 1088, 1089, 1090, 1094, 1106, 1107, 1111, 1125, 1127, 1128, 1138, 1140, 1151, 1152, 1154, 1160, 1163, 1164, 1169, 1170, 1175, 1176, 1179, 1180, 1182, 1184, 1185, 1194, 1195, 1196, 1200, 1203]

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def sigmoid_focal_loss_kd(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    targets_indicator = (targets != 0).long()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets_indicator + (1 - alpha) * (1 - targets_indicator)
        loss = alpha_t * loss

    return loss.mean(1).sum()

# def sigmoid_focal_loss_kd(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#     Returns:
#         Loss tensor
#     """
#     prob = inputs.sigmoid()
#     prob = prob.view(-1, prob.shape[2])
#     targets = targets.view(-1, targets.shape[2])
#     prob_norm = F.normalize((prob), dim=1)
#     targets_norm = F.normalize((targets), dim=1)
#
#     p_t = prob * targets + (1 - prob) * (1 - targets)
#     loss = p_t
#
#     return loss.mean(1).sum()

class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None], to_tuple(size), mode="bilinear").squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
