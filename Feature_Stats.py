import argparse
import yaml
import os                                
import numpy as np
import matplotlib.pyplot as plt

## <One line description> 
#  <Longer description>

def main(): 

    #------------#
    # Parameters #
    #------------#
    # Define script arguments
    parser = argparse.ArgumentParser(description='Compute feature statistics')
    parser.add_argument('-y', '--yaml-config', default='config/config_feature_homography.yaml', help='YAML config file')

    # Get arguments
    args = parser.parse_args()

    # Get YAML file
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Configure paths
    dataset_path = os.getcwd() + '/' + config['dir_dataset']
    results_path = os.getcwd() + '/' + config['dir_results']
    stats_path   = os.getcwd() + '/' + config['dir_stats']

    # Suppress scienfitic notation for numpy
    np.set_printoptions(suppress=True)

    # Output file
    output_file = stats_path + 'Output_Stats.npz'

    #----------------#
    # Data & Folders # 
    #----------------#

    # Get all timestamps 
    timestamps_list = [name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]

    print("-----------------")
    print("Num Images: " + str(len(timestamps_list)))
    print("-----------------")

    # =====================================================================


    #------------#
    # Initialize # 
    #------------#

    # SIFT
    time_sift_tot    = None
    err_pts_sift_tot = None
    err_pts_inlier_sift_tot  = None
    err_pts_outlier_sift_tot = None

    
    #--------------#
    # Collect Data # 
    #--------------#

    # Loop through all timestamps
    for timestamp in timestamps_list: 

        # Load data for timestamp
        with np.load(results_path + timestamp + '/Output-' + timestamp + '.npz') as ts_data:

            # SIFT
            if ts_data['err_avg_sift'] != -1: 
                time_sift    = ts_data['time_sift']
                err_pts_sift = ts_data['err_pts_sift']
                mask_sift    = ts_data['mask_sift']
                err_pts_inlier_sift  = err_pts_sift[mask_sift[:,0] != 0,:]
                err_pts_outlier_sift = err_pts_sift[mask_sift[:,0] == 0,:]

                if time_sift_tot is not None:
                    time_sift_tot            = np.concatenate((time_sift_tot,            np.array([time_sift])))
                    err_pts_sift_tot         = np.concatenate((err_pts_sift_tot,         err_pts_sift))
                    err_pts_inlier_sift_tot  = np.concatenate((err_pts_inlier_sift_tot,  err_pts_inlier_sift))
                    err_pts_outlier_sift_tot = np.concatenate((err_pts_outlier_sift_tot, err_pts_outlier_sift))

                else: 
                    time_sift_tot            = np.array([time_sift])
                    err_pts_sift_tot         = err_pts_sift
                    err_pts_inlier_sift_tot  = err_pts_inlier_sift
                    err_pts_outlier_sift_tot = err_pts_outlier_sift

                '''
                print(err_pts_sift.shape)
                print(err_pts_inlier_sift.shape)
                print(err_pts_outlier_sift.shape)
                '''


    #-----------#
    # Run Stats # 
    #-----------#

    #----- SIFT -----#
    # Time
    time_avg_sift = np.average(time_sift_tot)

    # Averages
    err_pts_avg_sift         = np.average(err_pts_sift_tot)
    err_pts_inlier_avg_sift  = np.average(err_pts_inlier_sift_tot)
    err_pts_outlier_avg_sift = np.average(err_pts_outlier_sift_tot)

    # Standard Deviation
    err_pts_std_dev_sift        = np.sqrt(np.var(err_pts_sift_tot))
    err_pts_inlier_std_dev_sift = np.sqrt(np.var(err_pts_inlier_sift_tot))

    #-------------#
    # Print Stats # 
    #-------------#
    print("time_avg_sift: "               + str(time_avg_sift))
    print("err_pts_avg_sift: "            + str(err_pts_avg_sift))
    print("err_pts_inlier_avg_sift: "     + str(err_pts_inlier_avg_sift))
    print("err_pts_outlier_avg_sift: "    + str(err_pts_outlier_avg_sift))
    print("err_pts_std_dev_sift: "        + str(err_pts_std_dev_sift))
    print("err_pts_inlier_std_dev_sift: " + str(err_pts_inlier_std_dev_sift))
    print("-----------------")

    #-----------#
    # Histogram # 
    #-----------#
    # Stacked histogram of inliers and outliers
    plt.hist([err_pts_inlier_sift_tot[:,0],err_pts_outlier_sift_tot[:,0]], 
                stacked=True,
                bins=100,
                color=["g","r"], # green inliers red outliers
                label=['Inliers','Outliers'],
                edgecolor='black',
            )
    plt.legend(prop={'size': 10})
    plt.xlabel("SIFT Matching Error (Euclidean Distance in Pixels)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Save image
    plt.savefig(stats_path+'Match_Error_Stacked_Hist_SIFT.png')

    # Show Image
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()



    # =====================================================================

    #------------#
    # Initialize # 
    #------------#

    # ORB
    time_orb_tot    = None
    err_pts_orb_tot = None
    err_pts_inlier_orb_tot  = None
    err_pts_outlier_orb_tot = None

    
    #--------------#
    # Collect Data # 
    #--------------#

    # Loop through all timestamps
    for timestamp in timestamps_list: 

        # Load data for timestamp
        with np.load(results_path + timestamp + '/Output-' + timestamp + '.npz') as ts_data:

            # ORB
            if ts_data['err_avg_orb'] != -1: 
                time_orb    = ts_data['time_orb']
                err_pts_orb = ts_data['err_pts_orb']
                mask_orb    = ts_data['mask_orb']
                err_pts_inlier_orb  = err_pts_orb[mask_orb[:,0] != 0,:]
                err_pts_outlier_orb = err_pts_orb[mask_orb[:,0] == 0,:]

                if time_orb_tot is not None:
                    time_orb_tot            = np.concatenate((time_orb_tot,            np.array([time_orb])))
                    err_pts_orb_tot         = np.concatenate((err_pts_orb_tot,         err_pts_orb))
                    err_pts_inlier_orb_tot  = np.concatenate((err_pts_inlier_orb_tot,  err_pts_inlier_orb))
                    err_pts_outlier_orb_tot = np.concatenate((err_pts_outlier_orb_tot, err_pts_outlier_orb))

                else: 
                    time_orb_tot            = np.array([time_orb])
                    err_pts_orb_tot         = err_pts_orb
                    err_pts_inlier_orb_tot  = err_pts_inlier_orb
                    err_pts_outlier_orb_tot = err_pts_outlier_orb

                '''
                print(err_pts_orb.shape)
                print(err_pts_inlier_orb.shape)
                print(err_pts_outlier_orb.shape)
                '''


    #-----------#
    # Run Stats # 
    #-----------#

    #----- ORB -----#
    # Time
    time_avg_orb = np.average(time_orb_tot)

    # Averages
    err_pts_avg_orb         = np.average(err_pts_orb_tot)
    err_pts_inlier_avg_orb  = np.average(err_pts_inlier_orb_tot)
    err_pts_outlier_avg_orb = np.average(err_pts_outlier_orb_tot)

    # Standard Deviation
    err_pts_std_dev_orb        = np.sqrt(np.var(err_pts_orb_tot))
    err_pts_inlier_std_dev_orb = np.sqrt(np.var(err_pts_inlier_orb_tot))

    #-------------#
    # Print Stats # 
    #-------------#
    print("time_avg_orb: "               + str(time_avg_orb))
    print("err_pts_avg_orb: "            + str(err_pts_avg_orb))
    print("err_pts_inlier_avg_orb: "     + str(err_pts_inlier_avg_orb))
    print("err_pts_outlier_avg_orb: "    + str(err_pts_outlier_avg_orb))
    print("err_pts_std_dev_orb: "        + str(err_pts_std_dev_orb))
    print("err_pts_inlier_std_dev_orb: " + str(err_pts_inlier_std_dev_orb))
    print("-----------------")

    #-----------#
    # Histogram # 
    #-----------#
    # Stacked histogram of inliers and outliers
    plt.hist([err_pts_inlier_orb_tot[:,0],err_pts_outlier_orb_tot[:,0]], 
                stacked=True,
                bins=100,
                color=["g","r"], # green inliers red outliers
                label=['Inliers','Outliers'],
                edgecolor='black',
            )
    plt.legend(prop={'size': 10})
    plt.xlabel("ORB Matching Error (Euclidean Distance in Pixels)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Save image
    plt.savefig(stats_path+'Match_Error_Stacked_Hist_ORB.png')

    # Show Image
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()



    # =====================================================================

    #------------#
    # Initialize # 
    #------------#

    # MFD
    time_mfd_tot    = None
    err_pts_mfd_tot = None
    err_pts_inlier_mfd_tot  = None
    err_pts_outlier_mfd_tot = None

    
    #--------------#
    # Collect Data # 
    #--------------#

    # Loop through all timestamps
    for timestamp in timestamps_list: 

        # Load data for timestamp
        with np.load(results_path + timestamp + '/Output-' + timestamp + '.npz') as ts_data:

            # MFD
            if ts_data['err_avg_mfd'] != -1: 
                time_mfd    = ts_data['time_mfd']
                err_pts_mfd = ts_data['err_pts_mfd']
                mask_mfd    = ts_data['mask_mfd']
                err_pts_inlier_mfd  = err_pts_mfd[mask_mfd[:,0] != 0,:]
                err_pts_outlier_mfd = err_pts_mfd[mask_mfd[:,0] == 0,:]

                if time_mfd_tot is not None:
                    time_mfd_tot            = np.concatenate((time_mfd_tot,            np.array([time_mfd])))
                    err_pts_mfd_tot         = np.concatenate((err_pts_mfd_tot,         err_pts_mfd))
                    err_pts_inlier_mfd_tot  = np.concatenate((err_pts_inlier_mfd_tot,  err_pts_inlier_mfd))
                    err_pts_outlier_mfd_tot = np.concatenate((err_pts_outlier_mfd_tot, err_pts_outlier_mfd))

                else: 
                    time_mfd_tot            = np.array([time_mfd])
                    err_pts_mfd_tot         = err_pts_mfd
                    err_pts_inlier_mfd_tot  = err_pts_inlier_mfd
                    err_pts_outlier_mfd_tot = err_pts_outlier_mfd

                '''
                print(err_pts_mfd.shape)
                print(err_pts_inlier_mfd.shape)
                print(err_pts_outlier_mfd.shape)
                '''


    #-----------#
    # Run Stats # 
    #-----------#

    #----- MFD -----#
    # Time
    time_avg_mfd = np.average(time_mfd_tot)

    # Averages
    err_pts_avg_mfd         = np.average(err_pts_mfd_tot)
    err_pts_inlier_avg_mfd  = np.average(err_pts_inlier_mfd_tot)
    err_pts_outlier_avg_mfd = np.average(err_pts_outlier_mfd_tot)

    # Standard Deviation
    err_pts_std_dev_mfd        = np.sqrt(np.var(err_pts_mfd_tot))
    err_pts_inlier_std_dev_mfd = np.sqrt(np.var(err_pts_inlier_mfd_tot))

    #-------------#
    # Print Stats # 
    #-------------#
    print("time_avg_mfd: "               + str(time_avg_mfd))
    print("err_pts_avg_mfd: "            + str(err_pts_avg_mfd))
    print("err_pts_inlier_avg_mfd: "     + str(err_pts_inlier_avg_mfd))
    print("err_pts_outlier_avg_mfd: "    + str(err_pts_outlier_avg_mfd))
    print("err_pts_std_dev_mfd: "        + str(err_pts_std_dev_mfd))
    print("err_pts_inlier_std_dev_mfd: " + str(err_pts_inlier_std_dev_mfd))
    print("-----------------")


    #-----------#
    # Histogram # 
    #-----------#
    # Stacked histogram of inliers and outliers
    plt.hist([err_pts_inlier_mfd_tot[:,0],err_pts_outlier_mfd_tot[:,0]], 
                stacked=True,
                bins=100,
                color=["g","r"], # green inliers red outliers
                label=['Inliers','Outliers'],
                edgecolor='black',
            )
    plt.legend(prop={'size': 10})
    plt.xlabel("MFD Matching Error (Euclidean Distance in Pixels)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Save image
    plt.savefig(stats_path+'Match_Error_Stacked_Hist_MFD.png')

    # Show Image
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()



    # =====================================================================

    #-----------#
    # SAVE DATA #
    #-----------#
    np.savez(output_file,
    # SIFT 
    time_avg_sift=time_avg_sift, 
    err_pts_avg_sift=err_pts_avg_sift, 
    err_pts_inlier_avg_sift=err_pts_inlier_avg_sift, 
    err_pts_outlier_avg_sift=err_pts_outlier_avg_sift,
    err_pts_std_dev_sift=err_pts_std_dev_sift,
    err_pts_inlier_std_dev_sift=err_pts_inlier_std_dev_sift,
    # ORB 
    time_avg_orb=time_avg_orb, 
    err_pts_avg_orb=err_pts_avg_orb, 
    err_pts_inlier_avg_orb=err_pts_inlier_avg_orb, 
    err_pts_outlier_avg_orb=err_pts_outlier_avg_orb,
    err_pts_std_dev_orb=err_pts_std_dev_orb,
    err_pts_inlier_std_dev_orb=err_pts_inlier_std_dev_orb,
    # MFD 
    time_avg_mfd=time_avg_mfd, 
    err_pts_avg_mfd=err_pts_avg_mfd, 
    err_pts_inlier_avg_mfd=err_pts_inlier_avg_mfd, 
    err_pts_outlier_avg_mfd=err_pts_outlier_avg_mfd,
    err_pts_std_dev_mfd=err_pts_std_dev_mfd,
    err_pts_inlier_std_dev_mfd=err_pts_inlier_std_dev_mfd,
    )



if __name__ == "__main__":
    main()