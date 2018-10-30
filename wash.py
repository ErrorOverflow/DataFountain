import numpy as np
import pandas as pd

loandata = pd.DataFrame(pd.read_csv('C:\\Users\\Buaa-Aladdin\\Downloads\\train_2.csv'));

dupset ={'service_type','is_mix_service','online_time','1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','many_over_bill','contract_type','contract_time','is_promise_low_consume','net_service','pay_times','pay_num','last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time','service2_caller_time','gender','age','complaint_level','former_complaint_num','former_complaint_fee','current_service'}

train_dup = loandata.drop_duplicates(subset=dupset, keep='first', inplace=False);

train_dup.to_csv('train_2_dup.csv', index=False);