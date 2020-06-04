from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.section import Section

if __name__ == '__main__':
    section_dir = '/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1/full_image_coordinates.txt'
    #section_dir = '/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images_small_test1/010_S10R1/full_image_coordinates.txt'
    section_num = 10
    conf_fname = '../conf/conf_example.yaml'
    processes_num = 8
    out_fname = './output_stitched_sec{}.json'.format(section_num)

    section = Section.create_from_full_image_coordinates(section_dir, section_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf, processes_num)
    stitcher.stitch_section(section) # will stitch and update the section

    # Save the transforms to file
    import json
    print('Writing output to: {}'.format(out_fname))
    section.save_as_json(out_fname)
#     img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
#     for img_fname, img, transform in zip(img_fnames, imgs, transforms):
#         # assumption: the output image shape will be the same as the input image
#         out_fname = os.path.join(out_path, os.path.basename(img_fname))
#         img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
#         cv2.imwrite(out_fname, img_transformed)


