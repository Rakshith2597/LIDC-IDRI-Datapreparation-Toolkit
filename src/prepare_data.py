import argparse
from prep_pack import visualize
from prep_pack import generate_slices
from prep_pack import create_folds
from prep_pack import generate_patches


def main(args):
     """
    Perform the selected action based on the command-line arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
	if args.genslice:
		generate_slices.extract_slices()
		generate_slices.generate_lungseg()

	elif args.createfolds:

		create_folds.positive_negative_classifier()
		dict_subset = create_folds.subset_classifier()
		create_folds.assign_folds(dict_subset)
		if args.additional:
			create_folds.create_balanced_dataset(additional=True)
		else:
			create_folds.create_balanced_dataset()


	elif args.genpatch:

		generate_patches.generate_patchlist(patchtype='positive')
		generate_patches.generate_positive_patch()
		generate_patches.generate_patchlist(patchtype='negative')
		generate_patches.generate_negative_patch()
		

	elif args.visualize:
		seriesuid = args.seriesuid
		slice_num = args.sliceno

		visualize.visualize_data(seriesuid,slice_num)

	else:
		print('Arguments not passed. Use -h for help')


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Select action to be performed')

	parser.add_argument('--genslice', default=False, action='store_true',
						help='To create slices from 3D volume')
	parser.add_argument('--createfolds', default=False, action='store_true',
						help='Split dataset into 10 folds')	
	parser.add_argument('--genpatch', default=False, action='store_true',
						help='To create patches from 2D slices')
	parser.add_argument('--visualize', default=False, action='store_true',
						help='Visualize any one of the slices')	
	parser.add_argument('--additional', default=False, action='store_true',
						help='Add additional slices')		
	parser.add_argument('--sliceno',
						help='Slice number to visualize')
	parser.add_argument('--seriesuid',
						help='Seriesuid of slice to visualize')

	args= parser.parse_args()

	main(args)