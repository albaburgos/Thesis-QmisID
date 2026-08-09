from pathlib import Path
import pandas as pd
import uproot
import awkward as ak
import vector
from pandas import DataFrame
from tqdm import tqdm
from iwpc.data_modules.pandas_directory_data_module_builder import PandasDirDataModuleBuilder
import numpy as np
import pandas as pd
import awkward as ak

def prep_truth_and_reco_data(fil):
    with uproot.open(fil) as f:
        truth_data = f['reco'].arrays([
            'truth_el_pt',
            'truth_el_eta',
            'truth_el_phi',
            'truth_el_charge',
            'truth_el_MCTC_isPrompt',
        ])
        reco_data = f['reco'].arrays([
            'el_charge',
            'el_pt_NOSYS',
            'el_eta',
            'el_phi',
        ])

    truth_data = truth_data[truth_data['truth_el_MCTC_isPrompt'] == 1]
    truth_vecs = vector.zip({
        'pt': truth_data['truth_el_pt'],
        'eta': truth_data['truth_el_eta'],
        'phi': truth_data['truth_el_phi'],
        'mass': 0,
    })
    reco_vecs = vector.zip({
        'pt': reco_data['el_pt_NOSYS'],
        'eta': reco_data['el_eta'],
        'phi': reco_data['el_phi'],
        'mass': 0,
    })
    deltaR = truth_vecs[:, :, None].deltaR(reco_vecs[:, None, :])
    is_matched = deltaR < 0.2
    min_dists = ak.argmin(deltaR, axis=-1)
    num_reco_matches = ak.sum(is_matched, axis=2)
    num_truth_matches = ak.sum(is_matched, axis=1)
    matched_reco = reco_data[min_dists]
    contains_double_match_mask = ak.max(num_truth_matches, axis=1) < 2
    print(f"Discarding {1 - ak.mean(contains_double_match_mask)}% events due to double truth match")
    truth_data = truth_data[num_reco_matches == 1][contains_double_match_mask]
    matched_reco = matched_reco[num_reco_matches == 1][contains_double_match_mask]

    return truth_data, matched_reco

import numpy as np
import pandas as pd
import awkward as ak

def prep_df_unlabelled(fil):
    truth_data, matched_reco = prep_truth_and_reco_data(fil)

    truth_data["is_flipped"] = truth_data["truth_el_charge"] != matched_reco["el_charge"]
    truth_data["pt_err"] = truth_data["truth_el_pt"] - matched_reco["el_pt_NOSYS"]
    truth_data["truth_el_q_over_pt"] = truth_data["truth_el_charge"] / truth_data["truth_el_pt"]

    mask = truth_data["is_flipped"] == 0
    mask_eta = np.abs(truth_data["truth_el_eta"]) < 2.5
    mask_pt = (truth_data["truth_el_pt"] > 5000) & (truth_data["truth_el_pt"] < 800000)

    sel = mask & mask_eta & mask_pt
    truth_data = truth_data[sel]
    matched_reco = matched_reco[sel]

    truth_pt  = ak.to_numpy(ak.flatten(truth_data["truth_el_pt"]))
    truth_eta = ak.to_numpy(ak.flatten(truth_data["truth_el_eta"]))
    pt_err    = ak.to_numpy(ak.flatten(truth_data["pt_err"]))

    reco_pt   = ak.to_numpy(ak.flatten(matched_reco["el_pt_NOSYS"]))
    reco_eta  = ak.to_numpy(ak.flatten(matched_reco["el_eta"]))

    finite_truth = np.isfinite(truth_pt) & np.isfinite(truth_eta) & np.isfinite(pt_err)
    finite_reco  = np.isfinite(reco_pt) & np.isfinite(reco_eta)

    df_truth = pd.DataFrame({
        "pt": truth_pt[finite_truth],
        "eta": truth_eta[finite_truth],
        "label": 0,
    })

    df_reco = pd.DataFrame({
        "pt": reco_pt[finite_reco],
        "eta": reco_eta[finite_reco],
        "label": 1,
    })

    return pd.concat([df_truth, df_reco], ignore_index=True)


def prep_df(fil):
    truth_data, matched_reco = prep_truth_and_reco_data(fil)

    truth_data["is_flipped"] = (
        truth_data["truth_el_charge"] != matched_reco["el_charge"]
    )

    truth_data["truth_el_q_over_pt"] = (
        truth_data["truth_el_charge"] / truth_data["truth_el_pt"]
    )

    matched_reco["reco_el_q_over_pt"] = (
        matched_reco["el_charge"] / matched_reco["el_pt_NOSYS"]
    )

    mask_not_flipped = truth_data["is_flipped"] == 1

    mask_eta = (
        (np.abs(truth_data["truth_el_eta"]) < 2.5) &
        (np.abs(matched_reco["el_eta"]) < 2.5)
    )

    mask_pt = (
        (truth_data["truth_el_pt"] > 5000) &
        (truth_data["truth_el_pt"] < 800000) &
        (matched_reco["el_pt_NOSYS"] > 5000) &
        (matched_reco["el_pt_NOSYS"] < 800000)
    )
    mask =  mask_not_flipped & mask_eta & mask_pt

    truth_data = truth_data[mask]
    matched_reco = matched_reco[mask]

    pt_err = -truth_data["truth_el_pt"] + matched_reco["el_pt_NOSYS"]
    q_over_pt_err = -truth_data["truth_el_q_over_pt"] + matched_reco["reco_el_q_over_pt"]

    truth_pt  = ak.to_numpy(ak.flatten(truth_data["truth_el_pt"]))
    truth_eta = ak.to_numpy(ak.flatten(truth_data["truth_el_eta"]))
    truth_q_pt = ak.to_numpy(ak.flatten(truth_data["truth_el_q_over_pt"]))
    pt_err_np = ak.to_numpy(ak.flatten(pt_err))
    pt_reco = ak.to_numpy(ak.flatten(matched_reco["el_pt_NOSYS"]))
    eta_reco = ak.to_numpy(ak.flatten(matched_reco["el_eta"]))
    label = ak.to_numpy(ak.flatten(truth_data["is_flipped"]))
    q_pt_err = ak.to_numpy(ak.flatten(q_over_pt_err))
    q_pt_err_rel = ak.to_numpy(ak.flatten(q_over_pt_err / truth_data["truth_el_q_over_pt"]))
    reco_q_pt = ak.to_numpy(ak.flatten(matched_reco["reco_el_q_over_pt"]))

    finite = np.isfinite(truth_pt) & np.isfinite(truth_eta) & np.isfinite(pt_err_np) & np.isfinite(pt_reco) & np.isfinite(eta_reco) & np.isfinite(q_pt_err)& np.isfinite(q_pt_err_rel)& np.isfinite(truth_q_pt)& np.isfinite(reco_q_pt)

    df_truth = pd.DataFrame({
        "pt": truth_pt[finite],
        "eta": truth_eta[finite],
        "q_pt": truth_q_pt[finite],
        "pt_err": pt_err_np[finite],
        "label": label[finite],
        "pt_reco": pt_reco[finite], 
        "eta_reco": eta_reco[finite],
        "q_pt_reco": reco_q_pt[finite],
        "q_pt_err": q_pt_err[finite],
        "q_pt_err_rel": q_pt_err_rel[finite],

    })

    return df_truth


def prep_df_dielectron(fil):
    truth_data, matched_reco = prep_truth_and_reco_data(fil)

    truth_data["is_flipped"] = (
        truth_data["truth_el_charge"] != matched_reco["el_charge"]
    )

    truth_data["truth_el_q_over_pt"] = (
        truth_data["truth_el_charge"] / truth_data["truth_el_pt"]
    )

    matched_reco["reco_el_q_over_pt"] = (
        matched_reco["el_charge"] / matched_reco["el_pt_NOSYS"]
    )

    mask_num = (
        (ak.num(matched_reco["el_pt_NOSYS"]) == 2) &
        (ak.num(truth_data["truth_el_pt"]) == 2)
    )

    mask_flipped = truth_data["is_flipped"] == 1

    mask_eta = (
        (np.abs(truth_data["truth_el_eta"]) < 2.5) &
        (np.abs(matched_reco["el_eta"]) < 2.5)
    )

    mask_pt = (
        (truth_data["truth_el_pt"] > 5000) &
        (truth_data["truth_el_pt"] < 800000) &
        (matched_reco["el_pt_NOSYS"] > 5000) &
        (matched_reco["el_pt_NOSYS"] < 800000)
    )

    mask = mask_num & ak.all(mask_flipped & mask_eta & mask_pt, axis=1)

    truth_data = truth_data[mask]
    matched_reco = matched_reco[mask]

    pt_err = -truth_data["truth_el_pt"] + matched_reco["el_pt_NOSYS"]
    q_over_pt_err = -truth_data["truth_el_q_over_pt"] + matched_reco["reco_el_q_over_pt"]

    def to_numpy_1d(arr):
        return ak.to_numpy(ak.fill_none(arr, np.nan))

    df = pd.DataFrame()
    for i in range(2):
        for k in matched_reco.fields:
            df[f'{k}_{i}'] = to_numpy_1d(matched_reco[k][:, i])
        for k in truth_data.fields:
            df[f'{k}_{i}'] = to_numpy_1d(truth_data[k][:, i])

        df[f"pt_err_{i}"] = to_numpy_1d(pt_err[:, i])
        df[f"q_pt_err_{i}"] = to_numpy_1d(q_over_pt_err[:, i])
        df[f"q_pt_err_rel_{i}"] = to_numpy_1d(
            q_over_pt_err[:, i] / truth_data["truth_el_q_over_pt"][:, i]
        )

    #numeric_cols = df.select_dtypes(include=[np.number]).columns
    #finite = np.isfinite(df[numeric_cols].to_numpy()).all(axis=1)
    #df = df.loc[finite].reset_index(drop=True)

    return df

def prep_reco_df(fil):
    with uproot.open(fil) as f:
        reco_data = f['reco'].arrays([
            'el_charge',
            'el_pt_NOSYS',
            'el_eta',
            'el_phi',
        ])
    reco_data = reco_data[ak.num(reco_data['el_pt_NOSYS']) == 2]
    reco_data['el_q_over_pt'] = reco_data['el_charge'] / reco_data['el_pt_NOSYS']
    df = pd.DataFrame()
    for i in range(2):
        for k in reco_data.fields:
            df[f'{k}_{i}'] = reco_data[k][:, i]

    df['label'] = df['el_charge_0'] == df['el_charge_1']

    return df


if __name__ == '__main__':

    root_dir = Path("user.egramsta.700788.Sh.DAOD_PHYSLITE.e8514_s4162_r15540_p6697.2L3LSUSYalphaV02_output")
    with PandasDirDataModuleBuilder(
        "/Users/albaburgosmondejar/QMISID/PtErrSC",
        force=True,
        file_size=int(5e4),
    ) as builder:
        for root_file in tqdm(list(root_dir.glob("*.root"))):
            df = prep_df(root_file)
            builder.write(df)
