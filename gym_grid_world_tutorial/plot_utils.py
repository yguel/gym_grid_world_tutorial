from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from matplotlib import pyplot as plt

@dataclass
class Params():
    savefig_folder: Path  # Root folder where plots are saved

params = Params(savefig_folder=Path("./img/step_by_step_tutorial/")
)

def set_savefig_folder(folder: Path):
    params.savefig_folder = folder

def plot_env(env,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.render())
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_env_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()

def plot_env_coords(env,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.render_with_coordinates())
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_env_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()

def plot_val(env,values,minmax,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.create_value_image(values,minmax))
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_values_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()

def plot_val_coords(env,values,minmax,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.create_value_image_with_row_col_coordinates(values,minmax))
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_values_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()

def plot_policy(env,policy,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.create_policy_image(policy))
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_policy_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()

def plot_policy_coords(env,policy,frame_title: Optional[str]=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.imshow(env.create_policy_image_with_row_col_coordinates(policy))
    ax.axis("off")
    if None != frame_title:
        ax.set_title("Frame "+frame_title)
        img_title = f"frozenlake_policy_frame_{frame_title}.png"
        fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    else:
        fig.savefig(params.savefig_folder, bbox_inches="tight")
    plt.show()