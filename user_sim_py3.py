#! /bin/bash/python

import numpy as np
import pandas as pd
import random
import json
import time
import math
import csv
from math import exp

random.seed(time.time())

def df_for_state(state_name):
    return full_pdb16_tr_df.loc[(full_pdb16_tr_df['State_name'] == state_name)]

#draw a random user, generate his profile, criteria
def draw_rand_user():
    usr_index = int(random.uniform(0, state_tr_df.shape[0]))
    #print state_tr_df.loc[usr_index]
    return state_tr_df.loc[usr_index]

# generate user criteria (old)
def usr_criteria_gen(user):
	# Total number of available attribute for user to select
	total_attr_count = len(attr_prop)

	# Required attribute: Price range
	required_attr_indices = [1,27,29,37,len(attr_prop)-1]

	# Aside from PriceRange, user also has m additional requirements
	# m is a random integer [required_attr + 1, total_attr_count]
	addi_attr_num = random.randint(len(required_attr_indices) + 1, total_attr_count)
	# index of additional attribute, might contain indices in required_attr
	addi_attr_indices = list(np.random.choice(total_attr_count, addi_attr_num))

	# required + additional
	usr_all_attr_indices = set(required_attr_indices + addi_attr_indices)

	bin_encode = 0
	explicit_filter = 0
	max_price_bucket = 0
	duplicate_na_one_bits = 0

	# generate selection for each attribute
	for i in range(0, len(attr_prop)):

		this_attr = attr_prop[i]

		all_one_bit_mask = (1 << (this_attr[0] + 1)) - 1 # '1' * this_attr[0]
		price_encoding = 0

		# if this is an attribute that the user cares about
		if i in usr_all_attr_indices:
			choice = random.randint(0, this_attr[0])

			## price range attribute
			if i == len(attr_prop)-1:
				try:
					usr_Income = float(user['avg_Agg_HH_INC_ACS_10_14'][1:].replace(',',''))
				except:
					print("Invalid Income, discard user")
					return None

				#print("usr_Income %f " % usr_Income)

				# Assume user's maximum tolarable price will be 5.2 times his income, see Reference
				try:
					# The real max tolerable price for this user
					max_price_bucket = next(x[0] for x in enumerate(house_price_bucket) if x[1] > 2.6 * usr_Income)
					# The applied tolerable price for this user
					filter_price_bucket = next(x[0] for x in enumerate(house_price_bucket) if x[1] > 3.5 * usr_Income)
					if filter_price_bucket == 0: return None
					this_attr_bit_encoding = all_one_bit_mask ^ ((1 << (this_attr[0] - filter_price_bucket)) - 1)
				except:
					this_attr_bit_encoding = all_one_bit_mask
				price_encoding = this_attr_bit_encoding
			else:
				this_attr_bit_encoding = 1 << choice

			##
		# if this is not an attribute that the user cares about
		else:
			this_attr_bit_encoding = all_one_bit_mask
			duplicate_na_one_bits = duplicate_na_one_bits + this_attr[0] - 1

		bin_encode = bin_encode << this_attr[0] | this_attr_bit_encoding


	#print("{0:b}".format(bin_encode))
	return (bin_encode, price_encoding, duplicate_na_one_bits)

# generate user criteria
def usr_criteria_gen_template(user):

	# discard user if income is invalid
	try:
		usr_Income = float(user['avg_Agg_HH_INC_ACS_10_14'][1:].replace(',',''))
	except:
		print("Invalid Income, discard user")
		return None

	# The real max tolerable price for this user
	try:
		max_price_bucket = next(x[0] for x in enumerate(house_price_bucket) if x[1] > 5.2 * usr_Income)
		if max_price_bucket == 0:
			print("No matched price range found, discard user")
			return None
	except:
		max_price_bucket = 18

	# if user income below certain level, no suitable price range can be found
	# discard user


	# Template matching
	this_usr_encode = selection_templates[max_price_bucket]

	# change all categorical variables to n/a
	for each in cat_attr_indices:
		this_cat_attribute_length = attr_prop[each][0]
		#starting position from the right (least significant bit)
		this_cat_starting_pos = attr_prop[each][2]

		# insert n/a cat_attr in the the binary string
		all_one_bit_mask = (1 << (this_cat_attribute_length)) - 1 # '1' * this_attr[0]
		this_usr_encode = replace_bits(this_cat_starting_pos, all_one_bit_mask, this_cat_attribute_length, this_usr_encode)

		# print("replacing n/a at pos %d, with length %d" % (this_cat_starting_pos, this_cat_attribute_length))

	allF = 0xFFFFF
	usr_price_encoding = ~((this_usr_encode & ((1 << 20) - 1)) - 1) & allF
	this_usr_encode = this_usr_encode | usr_price_encoding

	# print(("{0:b}".format(this_usr_encode),"{0:b}".format(usr_price_encoding)))
	return (this_usr_encode, usr_price_encoding)
'''
If usr_criteria_gen_template() was used, this method should be called
to randomize some numerical attributes so that users using the same template
do not have the same exact binary encoding.
'''
def numeric_attr_randomize(rand_pct, bin_encode):

	# the number of attr to randomize
	num_attr_to_rand = math.ceil(rand_pct * len(attr_prop))

	# sample certain numerical attribute
	# np.random.choice(Array, Size) draw a sample of size Size from Array
	attr_to_rand = set(np.random.choice(numeric_attr_indices[:-1], num_attr_to_rand))

	for each in attr_to_rand:
		attr_length = attr_prop[each][0]
		#starting position from the right (least significant bit)
		attr_starting_pos = attr_prop[each][2]

		choice = 1 << random.randint(0, attr_length)
		bin_encode = replace_bits(attr_starting_pos, choice, attr_length, bin_encode)

	return bin_encode


def insert_bits(starting_location, chunk_to_insert, length, bin_encode):
	#cleave current binary encode into two parts
	left = (1 << (starting_location + 1)) - 1 & bin_encode # less significant part
	right = bin_encode >> starting_location # more significant part

	bin_encode = right << (starting_location + length) | chunk_to_insert << starting_location | left

	return bin_encode


def replace_bits(starting_location, subsituting_chunk, length, bin_encode):

	all_one_bit_mask =  (1 << (length)) - 1

	# clear in-place the bits to be replaced
	bin_encode = ~(all_one_bit_mask << starting_location) & bin_encode
	bin_encode = bin_encode | (subsituting_chunk << starting_location)

	return bin_encode


def draw_houses_from_clusters(weights, filtered_clusters):

	houses_to_display = set()
	for i in range(1, display_size):

		# sample a cluster to draw from based on weights
		cluster_id = np.random.choice(sorted(clusters.keys()),p=list(weights))

		# draw an item from the selected cluster
		try:
			item = np.random.choice(filtered_clusters[cluster_id])
		except:
			i -= 1
			continue

		if item in houses_to_display:
			i -= 1
		else:
			houses_to_display.add(item)

	return houses_to_display

def init_price_filter(all_available_houses, price_encoding):

	filtered_clusters = {}
	for cluster_id in range(1, len(clusters.keys())):
		filtered_clusters[cluster_id] = []

	# Assuming each value in houses is a tuple (cluster_id, bin)
	# index by houseID
	total_qualified_houses = 0
	for houseID in all_available_houses.keys():
		house_bin_encode = int(all_houses[houseID][0])
		# check if houses price is within buyers' price range
		# print("house_bin_encode")
		# print("{0:b}".format(house_bin_encode))
		# print("price_encoding")
		# print("{0:b}".format(price_encoding))
		price_match = price_encoding & house_bin_encode
		if bin(price_match).count("1") == 1:
			try:
				filtered_clusters[all_houses[houseID][1]].append(houseID)
			except:
				filtered_clusters[all_houses[houseID][1]] = [houseID]
			total_qualified_houses += 1

	if total_qualified_houses < display_size:
		return None
	else:
		return filtered_clusters

# def top_houses()

'''
Draw user from the census tract data and generate user profile
user_bin_encode = binary encoding of user's critieria
clickable_items = a dictionary of items after initial filtering
return a triplet (user_county, user_bin_encode, clickable_item)

! This method could draw same user repeatedly !
'''
def draw_user_create_profile(randomize_pct = 0.1):

	user_Proxy = draw_rand_user()
	user_county = user_Proxy['County_name']

	# disgard user if exception raised due to invalid income
	try:
		usr_Income = float(user_Proxy['avg_Agg_HH_INC_ACS_10_14'][1:].replace(',',''))
	except:
		return draw_user_create_profile()

	# generate user's criteria based off user income and template selection
	usr_bin_encode, usr_price_encoding = usr_criteria_gen_template(user_Proxy)

	# randomize numerical attributes
	usr_bin_encode = numeric_attr_randomize(randomize_pct, usr_bin_encode)

	# Apply price filtering to only display houses below max tolerable price bucket
	filtered_clusters = init_price_filter(all_houses, usr_price_encoding)

	# If qualified housing items < display_size, redraw user
	if filtered_clusters == None:
		return draw_user_create_profile()

	return (user_county, usr_bin_encode, filtered_clusters)

# make one update per user
def EXP3_cluster_update(cluster_weight_by_county, see_past=True, gamma=0.1, alpha=0.4, iteration=10):
	this_usr_clicks = 0
	uniform_weight = np.array([1] * len(clusters.keys()))/float(len(clusters.keys()))
	if see_past and user_county in cluster_weight_by_county.keys():
		this_usr_weights = cluster_weight_by_county[user_county]
	else:
		this_usr_weights = uniform_weight

	for i in range(iteration):
		this_round = {}
		adj_cluster_weight = (1-gamma)*this_usr_weights+gamma*uniform_weight
		for eachHouse in draw_houses_from_clusters(adj_cluster_weight, filtered_clusters):
			# retrieve this house's bin_encoding and compute cosine similarity
			# between user's critieria
			house_bin_encode = house_item_info[eachHouse][0]
			house_belonging_cluster = house_item_info[eachHouse][1]
			cosine_sim = usr_bin_encode & house_bin_encode
			degree_of_sim = bin(cosine_sim).count("1")
			# if the house matches all critieria
			if len(attr_prop) * min_match_percentage <= degree_of_sim:
				this_usr_clicks += 1 # increment total user clicks
				# increment clicks of the belonging cluster
				this_round[house_belonging_cluster] = this_round.get(house_belonging_cluster, 0) + 1

		# update weight
		for group in this_round.keys():
			reward = this_round[group]/display_size/adj_cluster_weight[group]

			this_usr_weights[group] *= exp(-1.0*gamma*reward/len(clusters))
		this_usr_weights /= float(sum(this_usr_weights))

	if user_county not in cluster_weight_by_county.keys():
		cluster_weight_by_county[user_county] = this_usr_weights
	else:
		cluster_weight_by_county[user_county] = (1 - alpha) * \
					cluster_weight_by_county[user_county] + alpha * this_usr_weights
	return this_usr_clicks

# def EXP3_attribute_update(attribute_weight_by_county, see_past=True, gamma=0.1, alpha=0.4, iteration=10):
# 	this_usr_clicks = 0
# 	uniform_weight = np.array([1] * len(usr_bin_encode)/float(len(usr_bin_encode)))
# 	if see_past and user_county in attribute_weight_by_county.keys():
# 		this_usr_weights = attribute_weight_by_county[user_county]
# 	else:
# 		this_usr_weights = uniform_weight

# 	for i in range(iteration):
# 		this_round = {}
# 		adj_attri_weight = (1-gamma)*this_usr_weights+gamma*uniform_weight
# 		for eachHouse in draw_houses_from_clusters(adj_cluster_weight, filtered_clusters):
# 			# retrieve this house's bin_encoding and compute cosine similarity
# 			# between user's critieria
# 			house_bin_encode = house_item_info[eachHouse][0]
# 			house_belonging_cluster = house_item_info[eachHouse][1]
# 			cosine_sim = usr_bin_encode & house_bin_encode
# 			degree_of_sim = bin(cosine_sim).count("1")
# 			# if the house matches all critieria
# 			if len(attr_prop) * min_match_percentage <= degree_of_sim:
# 				this_usr_clicks += 1 # increment total user clicks
# 				# increment clicks of the belonging cluster
# 				this_round[house_belonging_cluster] = this_round.get(house_belonging_cluster, 0) + 1

# 		# update weight
# 		for group in this_round.keys():
# 			reward = this_round[group]/display_size/adj_cluster_weight[group]

# 			this_usr_weights[group] *= exp(-1.0*gamma*reward/len(clusters))
# 		this_usr_weights /= float(sum(this_usr_weights))

# 	if user_county not in cluster_weight_by_county.keys():
# 		cluster_weight_by_county[user_county] = this_usr_weights
# 	else:
# 		cluster_weight_by_county[user_county] = (1 - alpha) * \
# 					cluster_weight_by_county[user_county] + alpha * this_usr_weights
# 	return this_usr_clicks

# properties of attributes presented in a list of tuples
#			(#_of_choices, is_multiple_selection_allowed)
# pricing, rating: ex. price range 100k~1000k with 10 buckets => (10, True)
# unique feature set: ex. flooring type of wooden, carpet => (2, False)

attr_prop = [(3,False), (10,True),(2,False),(4,True),(5,False),(25,False),
(9,False),(5,False),(8,False),(10,True),(9,True),(10,True),(10,True),
(6,False),(15,False),(3,False),(10,True),(4,True),(6,False),(4,True),
(4,True),(4,True),(10,True),(5,True),(10,True),(4,True),(3,True),(4,True),
(3,True),(8,True),(4,True),(4,True),(5,True),(5,True),(6,False),(10,True),
(3,False),(5,True),(4,True),(4,False),(20,True)]

# include starting position (from the left most position) of each attribute
index = 0
new_attr_prop = []
for each in reversed(attr_prop):
	each_new = list(each)
	each_new.append(index)
	new_attr_prop = [each_new] + new_attr_prop
	index += each[0]
attr_prop = new_attr_prop

# categorical attributes indices
cat_attr_indices = [0,2,4,5,6,7,8,13,14,15,18,34,39]

price_range_attr_index = [40]

# all numeric attributes indices
numeric_attr_indices = list(set(range(0, len(attr_prop))).difference(cat_attr_indices))


house_price_bucket = [52301.2, 66702.4, 81103.6, 95504.8, 109906,
124307.2, 138708.4, 153109.6, 167510.8, 181912, 196313.2, 210714.4,
225115.6, 239516.8, 253918, 268319.2, 282720.4, 297121.6, 311522.8, 325924]


# Load datasets
default_encoding = "ISO-8859-1"
full_pdb16_tr_df = pd.read_csv("pdb2016trv8_us.csv", encoding=default_encoding)
state_tr_df = df_for_state( "Texas")
# Filter columns for relevant attributes
state_tr_df = state_tr_df[['County_name',
                                    'Tot_Population_ACS_10_14',
                                    'LAND_AREA', # density = total_pop/area
                                    'pct_Males_ACS_10_14',
                                    'pct_Females_ACS_10_14',
                                    'pct_Pop_under_5_ACS_10_14', #demographic breakdown
                                    'pct_Pop_18_24_ACS_10_14',
                                    'pct_Pop_25_44_ACS_10_14',
                                    'pct_Pop_45_64_ACS_10_14',
                                    'pct_Pop_65plus_ACS_10_14',
                                    'avg_Agg_House_Value_ACS_10_14',
                                    'pct_College_ACS_10_14',
                                    'avg_Agg_HH_INC_ACS_10_14'
                                   ]]

# delete all the n/a entries and reset index
state_tr_df = state_tr_df.reset_index()

# Read housing items info from csv file
clusters = {} # {cluster_id: [houseIDs]}
house_item_info = {} # {houseId: (bin_encode, belong_cluster)}

df = pd.read_csv("final_house9.csv")
for index in range(1, df.shape[0]):

	item_id = int(df.loc[index]['Id'])
	belonging_cluster = int(df.loc[index]['group'])

	#append housing item into clusters
	cluster_id = int(df.loc[index]['group'])
	if cluster_id in clusters.keys():
		clusters[cluster_id].append(item_id)
	else:
		clusters[cluster_id] = [item_id]

	# bin_encoding for each housing item
	bin_encode =  df.loc[index]['whole']
	bin_encode = bin_encode.replace(' ', '')
	bin_encode = bin_encode.replace('\n', '')
	bin_encode = bin_encode.replace('[', '')
	bin_encode = bin_encode.replace(']', '')
	bin_encode = int(bin_encode, 2)

	# Each house represented by a tuple of (house_bin_encode, belonging cluster)
	house_item_info[item_id] = (bin_encode, belonging_cluster)

# Read selection templates from csv file

selection_templates = {}
df = pd.read_csv("template_house.csv")
for index in range(0, df.shape[0]):
	# bin_encoding for each housing item
	bin_encode =  df.loc[index][1]
	bin_encode = bin_encode.replace(' ', '')
	bin_encode = bin_encode.replace('\n', '')
	bin_encode = bin_encode.replace('[', '')
	bin_encode = bin_encode.replace(']', '')
	bin_encode = int(bin_encode, 2)
	selection_templates[index+1] = bin_encode


num_sim_user = 1000
display_size = 10 # numer of houses to display each time
actual_sim_user = 0
min_match_percentage = 0.8 # require similarity between house item and user criteria

# Assuming each value in houses is a tuple (cluster_id, bin)
# index by houseID
all_houses = house_item_info

final_loc_result = []
final_no_loc_result = []
for i in range(0,55,5):
	alpha = i/100.0
	for j in range(0,20):
		gamma = j/100.0
		loc_clicks = 0
		loc_weights = {}
		no_loc_clicks = 0
		no_loc_weights = {}

		for u_id in range(num_sim_user):

			user_county, usr_bin_encode, filtered_clusters = draw_user_create_profile()

			loc_clicks += EXP3_cluster_update(cluster_weight_by_county=loc_weights, see_past=True, gamma=gamma, alpha=alpha, iteration=10)
			no_loc_clicks += EXP3_cluster_update(cluster_weight_by_county=no_loc_weights, see_past=False, gamma=gamma, alpha=alpha, iteration=10)
			actual_sim_user += 1

		final_loc_result += [float(loc_clicks/actual_sim_user)]
		final_no_loc_result += [float(no_loc_clicks/actual_sim_user)]
for item in final_loc_result:
	print(item)
for item in final_no_loc_result:
	print(item)
		# print(a, "w/ location: Average #clicks among %d simulated users are %f" % (actual_sim_user, float(loc_clicks/actual_sim_user)))
		# print(a, "w/o location: Average #clicks among %d simulated users are %f" % (actual_sim_user, float(no_loc_clicks/actual_sim_user)))

# REFERENCE

# National Average house price to household income ratio: 2.6
# https://www.citylab.com/equity/2018/05/where-the-house-price-to-income-ratio-is-most-out-of-whack/561404/