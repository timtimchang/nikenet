from os import walk
from os.path import dirname, join, abspath
import csv
from detectboundingbox import generateBoundingBoxAndSave


def prepareAndSaveLabels(in_img_dir, out_img_dir, out_labels, out_categories):
    categories = []

    with open(out_labels, 'w') as labels_file:
        spamwriter = csv.writer(
            labels_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        headers = ['filename', 'category', 'bounding_box_x',
                   'bounding_box_y', 'bounding_box_width', 'bounding_box_height']
        idx = 0

        spamwriter.writerow(headers)

        for dirpath, dirs, files in walk(in_img_dir):
            if len(files) > 0:
                print('Processing...', dirpath)

                categs = dirpath.split('/')
                f_categs = ''
                cur_dir = 1

                while categs[-cur_dir] != 'images':
                    f_categs = categs[-cur_dir] + ' ' + f_categs
                    cur_dir += 1

                f_categs.strip()

                if len(f_categs) > 0:
                    try:
                        current_categ = categories.index(f_categs)
                    except ValueError:
                        current_categ = len(categories)
                        categories.append(f_categs)

                    current_categ += 1

                    for file in files:
                        if '.gitignore' != file:
                            idx += 1
                            new_filename, x, y, w, h = generateBoundingBoxAndSave(
                                join(dirpath, file), out_img_dir, str(idx))
                            spamwriter.writerow(
                                [new_filename, current_categ, x, y, w, h])

    with open(out_categories, 'w') as categs_file:
        c_file_writer = csv.writer(
            categs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for categ_idx in range(0, len(categories)):
            c_file_writer.writerow([(categ_idx + 1), categories[categ_idx]])


if __name__ == '__main__':
    # dest_dir = join(dirname(dirname(dirname(abspath("./")))), 'dest')
    dest_dir = "./dest/"
    #img_dir = join(dirname(dirname(dirname(abspath("./")))), 'images')
    img_dir = "./images/"

    dest_img_dir = join(dest_dir, 'downloads')
    dest_labels = join(dest_dir, 'shoesdataset.csv')
    dest_categories = join(dest_dir, 'shoescategories.csv')

    prepareAndSaveLabels(img_dir, dest_img_dir, dest_labels, dest_categories)
