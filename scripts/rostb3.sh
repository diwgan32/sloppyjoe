export ROS_MASTER_URI=http://veggie.local:11311
export ROS_IP=$(ifconfig ens33 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')
export ROS_HOSTNAME=$(ifconfig ens33 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')
