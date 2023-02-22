import logging
import os
import pickle
import time

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from census_datasets import ACSIncome_categories
from folktables import (ACSDataSource, ACSEmployment, ACSIncome,
                        ACSPublicCoverage, generate_categories)

STATE_LIST = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


def get_raw_data_by_client(state, args, survey_year="2018"):
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=args.data_cache_dir + "/%s/%s" % (survey_year, "1-Year"),
    )
    definition_df = data_source.get_definitions(download=True)
    public_categories = generate_categories(
        features=ACSPublicCoverage.features, definition_df=definition_df
    )
    employment_categories = generate_categories(
        features=ACSEmployment.features, definition_df=definition_df
    )

    acs_data = data_source.get_data(states=[state], download=True)
    if args.task == "employment":
        x, y, s = ACSEmployment.df_to_pandas(
            acs_data, categories=employment_categories, dummies=True
        )
        x, y, s = x.to_numpy(), y.to_numpy(), s.to_numpy()
        print(x.shape)
    elif args.task == "income" and args.attribute == "race":
        start_time = time.time()
        x, y, s = ACSIncome.df_to_pandas(
            acs_data, categories=ACSIncome_categories, dummies=True
        )
        x, y, s = x.to_numpy(), y.to_numpy(), s.to_numpy()
        print(time.time() - start_time)

    elif args.task == "health":
        x, y, s = ACSPublicCoverage.df_to_pandas(
            acs_data, categories=public_categories, dummies=True
        )
        x, y, s = x.to_numpy(), y.to_numpy(), s.to_numpy()
        print(x.shape)
    return x, y, s


def partition_dataset(y, args):
    all_index = [i for i in range(y.shape[0])]
    num_train = int(args.partition.split("_")[0])
    num_test = int(args.partition.split("_")[1])
    num_val = int(args.partition.split("_")[2])
    r_train = num_train / (num_test + num_train + num_val)
    r_test = num_test / (num_test + num_train + num_val)
    r_val = num_val / (num_test + num_train + num_val)

    if len(all_index) < num_train + num_test + num_val:
        num_train = int(len(all_index) * r_train)
        num_test = int(len(all_index) * r_test)
        num_val = int(len(all_index) * r_val)

    s_train, s_all_test = train_test_split(
        all_index, train_size=int(num_train), random_state=args.random_seed
    )
    s_test, s_val = train_test_split(
        s_all_test, train_size=int(num_test), random_state=args.random_seed
    )

    unselected_index = [
        i for i in all_index if i not in s_train and i not in s_test and i not in s_val
    ]

    return s_train, s_test, s_val, unselected_index


def get_dataloader(client_idx, args=None):
    task = args.task
    random_seed = args.random_seed
    state = STATE_LIST[client_idx]
    x, y, s = get_raw_data_by_client(state, args)

    train_index, test_index, val_index, unselected_index = partition_dataset(y, args)

    sc = StandardScaler()
    x = sc.fit_transform(x)
    le = LabelEncoder()
    y = le.fit_transform(y.ravel())

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x[test_index], dtype=torch.float),
        torch.tensor(y[test_index], dtype=torch.long),
        torch.tensor(s[test_index], dtype=torch.long),
    )

    validation_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x[val_index], dtype=torch.float),
        torch.tensor(y[val_index], dtype=torch.long),
        torch.tensor(s[val_index], dtype=torch.long),
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x[train_index], dtype=torch.float),
        torch.tensor(y[train_index], dtype=torch.long),
        torch.tensor(s[train_index], dtype=torch.long),
    )

    unselected = {
        "x": x[unselected_index],
        "y": y[unselected_index],
        "s": s[unselected_index],
        "num": len(s[unselected_index]),
    }

    return train_dataset, test_dataset, validation_dataset, unselected


def load_partition_data_census(users, args):
    filepath = "{}/data.pkl".format(args.run_folder)
    logging.info(filepath)
    if args.load_dataset and os.path.isfile(filepath):
        with open(filepath, "rb") as f: 
            dataset = pickle.load(f)

        return dataset

    else:
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        train_data_local_num_dict = dict()
        test_data_local_num_dict = dict()
        val_data_local_dict = dict()

        train_data_global_dataset = list()
        test_data_global_dataset = list()
        val_data_global_dataset = list()
        unselected_data_local_dict = dict()
        train_data_num = 0
        test_data_num = 0

        for client_idx in users:  # only for those users
            (
                train_dataset_local,
                test_dataset_local,
                val_dataset_local,
                unselected,
            ) = get_dataloader(client_idx, args)

            train_data_global_dataset.append(train_dataset_local)
            test_data_global_dataset.append(test_dataset_local)
            val_data_global_dataset.append(val_dataset_local)

            train_num = len(train_dataset_local)
            test_num = len(test_dataset_local)

            train_data_num += train_num
            test_data_num += test_num

            train_data_local_num_dict[client_idx] = train_num
            test_data_local_num_dict[client_idx] = test_num

            logging.info(
                "client_idx = %d, local_trainig_sample_number = %d, local_test_sample_number = %d"
                % (client_idx, len(train_dataset_local), len(test_dataset_local))
            )

            train_data_local_dict[client_idx] = torch.utils.data.DataLoader(
                train_dataset_local,
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
            test_data_local_dict[client_idx] = torch.utils.data.DataLoader(
                test_dataset_local,
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
            )
            val_data_local_dict[client_idx] = torch.utils.data.DataLoader(
                val_dataset_local,
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
            )
            unselected_data_local_dict[client_idx] = unselected

        train_data_global_dataset = torch.utils.data.ConcatDataset(
            train_data_global_dataset
        )
        test_data_global_dataset = torch.utils.data.ConcatDataset(
            test_data_global_dataset
        )
        val_data_global_dataset = torch.utils.data.ConcatDataset(
            val_data_global_dataset
        )

        train_data_global = torch.utils.data.DataLoader(
            train_data_global_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_data_global = torch.utils.data.DataLoader(
            test_data_global_dataset, batch_size=args.batch_size, shuffle=False
        )
        val_data_global = torch.utils.data.DataLoader(
            val_data_global_dataset, batch_size=args.batch_size, shuffle=False
        )

        class_num = 2

        dataset = [
            len(users),
            users,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            val_data_global,
            train_data_local_num_dict,
            test_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            class_num,
            unselected_data_local_dict,
        ]

        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)

        return (
            len(users),
            users,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            val_data_global,
            train_data_local_num_dict,
            test_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            class_num,
            unselected_data_local_dict,
        )
