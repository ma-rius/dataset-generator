# # import numpy as np
# #
# # for meta in complexities:
# #     # iterate through SUB complexities
# #     for sub_j in complexities.keys():
# #         # get meta data
# #         meta_data = pd.read_csv(
# #             filepath_or_buffer='../assets/complexity_%r/data_%r.csv' % (complexities[sub_j], (iter + 1)))
# #         meta_mst = create_dataset_and_or_mst(data=meta_data,
# #                                              path='../assets/complexity_%r/data_%r.csv' % (
# #                                                  complexities[sub_j], (iter + 1)))
# #         for k in range(3):
# #             # get sub data
# #             sub_data = pd.read_csv(filepath_or_buffer='../assets/complexity_%r/data_%r_%r.csv' % (
# #                 complexities[sub_j], (iter + 1), (k + 1)))
# #             sub_mst = create_dataset_and_or_mst(data=sub_data,
# #                                                 path='../assets/complexity_%r/data_%r_%r.csv' % (
# #                                                     complexities[sub_j], (iter + 1), (k + 1)))
# #
# #
# #             #     print('../assets/complexity_%r/data_%r_%r.csv' % (complexities[sub_j], (iter + 1), (k + 1)))
# #             # print('../assets/complexity_%r/data_%r.csv' % (complexities[sub_j], (iter + 1)))
# #             # print()
#
# for iter in range(20):
#     excel_sheet = excel_file.create_sheet(title='results')
#     # header
#     excel_sheet.cell(column=1, row=1, value='Csub')
#     excel_sheet.cell(column=2, row=1, value='Cmeta')
#     excel_sheet.cell(column=3, row=1, value='Ccomplete')
#     print('ITERATION')
#
#     for meta in complexity:
#         # get meta data:
#         meta_data = pd.read_csv(filepath_or_buffer='../assets/complexity_%r/data_%r_labels.csv' % (meta, (iter + 1)),
#                                 usecols=[0, 1, 2, 3, 4, 'label'])
#         print(meta_data)
#         meta_mst = create_dataset_and_or_mst(data=meta_data, path='../assets_/complexity_%r/data_%r_labels.csv' % (meta, (iter + 1)))
#         print(meta_data['label'])
#         C_meta = evaluate(meta_data['label'], meta_data, n=1000)
#         for sub in complexity:
#             for k in range(1, 4, 1):
#                 sub_k_data = pd.read_csv(filepath_or_buffer='../assets/complexity_%r/data_%r_%r.csv' % (
#                     (meta, (iter + 1), k)))
#                 sub_k_mst = create_dataset_and_or_mst(data=sub_k_data, path='../assets_/complexity_%r/data_%r_%r.csv' % (
#                     (meta, (iter + 1), k)))
#                 C_sub = evaluate(sub_k_data['label'], sub_k_mst, n=1000)
#         complete_data = pd.read_csv(filepath_or_buffer='../assets/complexity_%r/data_%r.csv' % (meta, (iter + 1)))
#         complete_mst = create_dataset_and_or_mst(data=complete_data,
#                                              path='../assets_/complexity_%r/data_%r.csv' % (meta, (iter + 1)))
#         C_complete = evaluate(complete_data['label'], complete_data, n=1000)
#
# for k in range(1,4,1):
#     print(k)
#
