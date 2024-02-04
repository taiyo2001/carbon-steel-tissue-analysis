import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_input_label_pre_image(input, label, predicted, name):
    gray_label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    gray_predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)

    figure, ax = plt.subplots(ncols=3, figsize=(15, 18))

    ax[0].imshow(input)
    ax[1].imshow(gray_label)
    ax[2].imshow(gray_predicted)
    ax[0].set_title(f"Input Image {name}")
    ax[1].set_title(f"Label Mask {name}")
    ax[2].set_title(f"Predicted Mask {name}")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.show()


def exec_ferrite_analysis(ferrite_image):
    gray_ferrite_image = cv2.cvtColor(ferrite_image, cv2.COLOR_BGR2GRAY)

    # 二値化処理を行う
    _, binary_image = cv2.threshold(gray_ferrite_image, 128, 255, cv2.THRESH_BINARY)
    # 輪郭を検出
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # 輪郭の個数
    contour_count = len(contours)

    # 各輪郭の周の長さと面積を取得
    contour_lengths = [cv2.arcLength(contour, closed=True) for contour in contours]
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # 結果を描画
    result_image = np.zeros_like(ferrite_image)
    cv2.drawContours(result_image, contours, -1, 255, thickness=cv2.FILLED)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (0, 255, 255)  # 青色
    font_thickness = 1

    num_columns = 5
    num_rows = (contour_count + num_columns - 1) // num_columns

    # plt.figure(figsize=(10, 10))
    plt.figure(figsize=(15, 15))

    for i, contour in enumerate(contours):
        # 輪郭の中心座標を計算
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 各輪郭を描画（青で塗りつぶす）
        cv2.drawContours(result_image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(result_image)
        plt.axis("off")
        plt.title(f"Contour {i+1}")

        # 各輪郭を描画（白色で塗りつぶす）
        cv2.drawContours(
            result_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
        )
        # 番号を輪郭の中心に配置
        cv2.putText(
            result_image,
            str(i + 1),
            (cX, cY),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    plt.tight_layout()
    plt.show()

    sum_areas = 0
    for area in contour_areas:
        sum_areas += area

    print(f"輪郭の個数: {contour_count}")
    print(f"各輪郭の周の長さ: {contour_lengths}")
    print(f"各輪郭の面積: {contour_areas}")
    print(f"各輪郭の面積(合計): {sum_areas}")

    plt.figure(figsize=(12, 12))
    plt.imshow(result_image)
    plt.show()


def exec_perlite_analysis(perlite_image):
    gray_perlite_image = cv2.cvtColor(perlite_image, cv2.COLOR_BGR2GRAY)

    # 二値化処理を行う
    _, binary_image = cv2.threshold(gray_perlite_image, 128, 255, cv2.THRESH_BINARY)

    binary_image = cv2.bitwise_not(binary_image)
    eroded_image = cv2.GaussianBlur(binary_image, (3, 3), 0)

    # さらに二値化
    _, eroded_image = cv2.threshold(eroded_image, 200, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(
        eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 面積が閾値未満の輪郭を削除
    min_contour_area = 3
    filtered_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_contour_area
    ]

    # filtered_contours後画像
    filtered_contours_image = np.zeros_like(perlite_image)
    cv2.drawContours(
        filtered_contours_image, filtered_contours, -1, 255, thickness=cv2.FILLED
    )

    _, filtered_contours_image = cv2.threshold(
        filtered_contours_image, 128, 255, cv2.THRESH_BINARY
    )

    # 輪郭の個数
    contour_count = len(filtered_contours)
    # 各輪郭の周の長さと面積を取得
    contour_lengths = [
        cv2.arcLength(contour, closed=True) for contour in filtered_contours
    ]
    contour_areas = [cv2.contourArea(contour) for contour in filtered_contours]
    # 結果を描画
    result_image = np.zeros_like(perlite_image)
    cv2.drawContours(result_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (0, 255, 255)  # 青色
    font_thickness = 1

    for i, contour in enumerate(filtered_contours):
        # 輪郭の中心座標を計算
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 番号を輪郭の中心に配置
        cv2.putText(
            result_image,
            str(i + 1),
            (cX, cY),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    sum_areas = 0
    for area in contour_areas:
        sum_areas += area

    print(f"輪郭の個数: {contour_count}")
    print(f"各輪郭の周の長さ: {contour_lengths}")
    print(f"各輪郭の面積: {contour_areas}")
    print(f"各輪郭の面積(合計): {sum_areas}")

    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.show()
