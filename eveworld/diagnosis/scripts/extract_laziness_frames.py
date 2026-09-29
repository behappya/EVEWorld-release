#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Laziness-evidence frame extraction: right half of side-by-side eval mp4s -> frame strip + montage.

Modes: --survey (uniform sample), --id (run a REGISTRY case), --video + --frames/--times (ad-hoc), --list/--all (batch).
"""

import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GAGI_ROOT = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
GAGIBENCH_ROOT = os.environ.get("GAGIBENCH_ROOT", os.path.expanduser("~/gagibench"))


# REGISTRY: hand-picked timestamps for reproduction.
# key = "<model>/<bench>_<dur>_<task>"; frames = right-half frame indices (4-6 ints).
# Try combinations in mode 3 (--video --frames), then paste the printed snippet here.
REGISTRY = {
    # Example case (frames filled in after a survey; see the note below)
    "gigaworld0/dreamgen_98_1": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
                 "to_bottom_level_of_the_shelf.mp4",
        "frames": [0, 20, 55, 90, 156],
        "note": "Instruction: move the peach from the top shelf to the bottom shelf. idx0 bottom "
                "tray empty; idx20(1.25s) peach already sits on the bottom tray = goal state, no "
                "grasp, no transport (teleport + early finish); idx55(3.44s) peach still on the "
                "bottom shelf, the arm only now approaches from the left (final state before the "
                "action); idx90(5.62s) arm hovers next to the peach without grasping (waving at "
                "nothing afterwards); idx156(9.75s) end: peach still in place, the transport never "
                "really happens. Typical Model Laziness: final state far earlier than any "
                "necessary action.",
    },
    "gigaworld0/dreamgen_58_1": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
                 "to_bottom_level_of_the_shelf.mp4",
        "frames": [0, 14, 28, 65, 92],
        "note": "Same instruction (peach top->bottom) at 5.8s. Laziness shows as objects spawning "
                "from nothing + count non-conservation: idx0(0.00s) top shelf empty, no peach; "
                "idx14(0.88s) two pink peaches appear out of nowhere (beside the gripper + in the "
                "background), not produced by grasping; idx28(1.75s) one large peach appears "
                "directly between the grippers, a second peach still drifts in the background "
                "(multi-object hallucination); idx65(4.06s) peach hovers between the grippers, the "
                "ghost peach still in the background; idx92(5.75s) final state: one peach lands on "
                "the bottom tray = target position, but a person top right still holds a "
                "hallucinated peach, no legitimate 'grasp on the top shelf -> transport -> place' "
                "process at any point. Typical Model Laziness: generates the final state directly, "
                "with object hallucinations.",
    },

    # Rubik's-cube task (bottom -> top), same prompt, three durations, laziness mode differs
    "gigaworld0/dreamgen_98_3": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 16, 55, 78, 156],
        "note": "Instruction: move the rubik's cube from the bottom level to the top level. "
                "Teleport + early finish: idx0 cube on the bottom level (start correct); "
                "idx16(1.0s) cube already teleported to the top-level target, held by the right "
                "gripper; a real bottom->top move is impossible within 1s (teleport + early "
                "finish); idx55(3.44s) a second red cube pops up top right (hallucination); "
                "idx78(4.88s) two cubes coexist in the top area (count non-conservation at its "
                "clearest); idx156(9.75s) end: cube on the top level, bottom level empty.",
    },
    "gigaworld0/dreamgen_58_3": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 19, 33, 65, 92],
        "note": "Same instruction, 5.8s run. Final state flashes then vanishes + never finishes: "
                "idx0 two cubes at the start (a green cube already on the top level = target "
                "position taken, plus one on the bottom level), abnormal start; idx19(1.19s) still "
                "one top and one bottom, grippers motionless; idx33(2.06s) the top cube "
                "disappears, leaving one in the middle of the bottom level (state jump); "
                "idx92(5.75s) end: only the bottom one left, top level empty = the task actually "
                "failed. The final state appears once then vanishes, compounding the count "
                "non-conservation.",
    },
    "gigaworld0/dreamgen_158_3": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_15p8s_full_20260627_145209/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 51, 126, 190, 252],
        "note": "Same instruction, 15.8s run. Two-cube hallucination throughout: idx0 two cubes at "
                "the start (colored one on top = target position taken + green one below); the "
                "real start should have had only the bottom one; idx51(3.19s) top cube disappears, "
                "only the green one below; idx126(7.88s) flips back to colored top + green bottom "
                "coexisting; idx252(15.75s) end: both cubes still coexist, the top cube propped by "
                "a human hand the whole time. Count never conserved, no legitimate 'grasp below -> "
                "move to the top' process.",
    },

    # Purple-glass task (blue plate -> teal plate), same prompt, three durations
    # Shared by all three: the blue plate starts with a short red cup, no purple glass; the glass
    # teleports straight onto the target teal plate and coexists with the red cup throughout,
    # count non-conservation. Wrong start + teleport to the final state + early finish.
    "gigaworld0/dreamgen_98_4": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 14, 30, 50, 70],
        "note": "Instruction: left hand moves the tall purple glass from the blue plate to the "
                "teal plate. idx0(0.00s) the left blue plate holds a short red cup, not the purple "
                "glass (wrong start, no purple glass on the blue plate), middle teal plate / right "
                "pink plate empty; idx31(1.94s) the tall purple glass appears from nothing on the "
                "middle teal plate = goal state, it never appeared on the blue plate and was never "
                "grasped or carried (teleport + early finish), red cup still on the blue plate; "
                "idx55(3.44s) glass steady on the teal plate, no 'grasp from the blue plate' "
                "anywhere; idx156(9.75s) end: glass on the teal plate, red cup still on the blue "
                "plate, both coexist, count non-conservation.",
    },
    "gigaworld0/dreamgen_58_4": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 8, 22, 55, 60],
        "note": "Same instruction, 5.8s run. Wrong start + teleport + shape-shift hallucination: "
                "idx0 short red cup on the left blue plate, no purple glass; idx33(2.06s) the tall "
                "purple glass appears from nothing on the middle teal plate (target position), "
                "while a distorted glass jar pops up on the right, red cup still on the blue "
                "plate; idx92(5.75s) end: purple glass on the teal plate + malformed glass vessel "
                "on the right + red cup on the blue plate, three containers coexisting, count "
                "non-conservation. The glass is generated straight on the target plate, never "
                "grasped or carried.",
    },
    "gigaworld0/dreamgen_158_4": {
        "video": f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_15p8s_full_20260627_145209/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 89, 177, 220, 252],
        "note": "Same instruction, 15.8s run (highest resolution, good for the main figure). idx0 "
                "short red cup on the left blue plate, middle teal plate / right pink plate empty, "
                "no purple glass at the start; idx89(5.56s) the tall purple glass is already on "
                "the middle teal plate = target position, the right gripper just reaches it, red "
                "cup still on the blue plate (glass generated straight at the endpoint, never "
                "grasped and carried from the blue plate); idx177(11.06s) glass steady on the teal "
                "plate, both containers coexisting; idx252(15.75s) end: glass still on the teal "
                "plate + red cup still on the blue plate, count non-conservation.",
    },
}


# New this round: DreamGen tasks auto-expand into REGISTRY as task -> model -> duration.
# To refine frames for an auto-expanded case later, write the same key into REGISTRY above;
# setdefault below will not overwrite hand-written entries.
DREAMGEN_SIDE_BY_SIDE_ROOT = (
    f"{GAGI_ROOT}/gr1_dreamgen_eval/generated_side_by_side"
)

DREAMGEN_SWEEP_RUNS_3P8_5P8_7P8 = [
    ("gigaworld0_pretrain", "38",
     "pretrain_dreamgen_8gpu_3p8s_full_short_3p8_7p8"),
    ("gigaworld0_pretrain", "58",
     "pretrain_dreamgen_8gpu_5p8s_full_20260627_162849"),
    ("gigaworld0_pretrain", "78",
     "pretrain_dreamgen_8gpu_7p8s_full_short_3p8_7p8"),
    ("gigaworld0_gr1_sft", "38",
     "sft_dreamgen_8gpu_3p8s_full_short_3p8_7p8"),
    # The 5.8s GR1/SFT run uses an early directory name; no explicit sft/5p8s.
    ("gigaworld0_gr1_sft", "58",
     "gr1_dreamgen_8gpu_full_20260625_212933"),
    ("gigaworld0_gr1_sft", "78",
     "sft_dreamgen_8gpu_7p8s_full_short_3p8_7p8"),
]

DREAMGEN_DEFAULT_FRAMES_BY_DUR = {
    "38": [0, 15, 30, 45, 60],
    "58": [0, 23, 46, 69, 92],
    "78": [0, 31, 62, 93, 124],
}

# Per-video frame/note refinements go here. The key is the REGISTRY key,
# so videos of the same duration can be tuned independently.
# Example:
#     "gigaworld0_pretrain/dreamgen_58_86": {
#         "frames": [0, 12, 28, 52, 76],
#         "note": "clear orange cup 5.8s pretrain; frames chosen to avoid the human hand.",
#     },
DREAMGEN_CASE_OVERRIDES = {
    # task 001: pink peach top shelf -> bottom shelf
    # gigaworld0_pretrain/dreamgen_58_1: usable per annotation; laziness then "no idea what to do next"
    "gigaworld0_gr1_sft/dreamgen_38_1": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_1": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_1": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_1": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_1": {"frames": [0, 23, 42, 72, 92]},
    "gigaworld0_pretrain/dreamgen_78_1": {"frames": [0, 31, 62, 93, 124]},

    # task 004: tall purple glass blue plate -> teal plate
    # usable per annotation: gigaworld0_pretrain/dreamgen_58_4; laziness then "no idea what to do"
    "gigaworld0_gr1_sft/dreamgen_38_4": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_4": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_4": {"frames": [0, 23, 58, 69, 79,91]},
    "gigaworld0_pretrain/dreamgen_38_4": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_4": {"frames": [0, 23, 58, 69, 79,91]},
    "gigaworld0_pretrain/dreamgen_78_4": {"frames": [0, 31, 62, 93, 124]},

    # task 005: green apple bottom shelf -> top shelf
    # usable per annotation: gigaworld0_gr1_sft/dreamgen_58_5
    "gigaworld0_gr1_sft/dreamgen_38_5": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_5": {"frames": [0, 23, 46, 50, 60,70]},
    "gigaworld0_gr1_sft/dreamgen_78_5": {"frames": [0, 20, 30,39, 40,124]},
    "gigaworld0_pretrain/dreamgen_38_5": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_5": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_5": {"frames": [0, 31, 62, 93, 124]},

    # task 008: green cucumber beige place mat -> small cyan plate 
    # left/right hands swapped
    "gigaworld0_gr1_sft/dreamgen_38_8": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_8": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_8": {"frames": [0, 15, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_8": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_8": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_8": {"frames": [0, 31, 62, 93, 124]},

    # task 014: rubik's cube bottom brown wooden shelf -> top shelf
    # usable per annotation, "no idea what to do next": gigaworld0_gr1_sft/dreamgen_58_14
    "gigaworld0_gr1_sft/dreamgen_38_14": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_14": {"frames": [0, 35, 50, 60, 67,92]},
    "gigaworld0_gr1_sft/dreamgen_78_14": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_14": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_14": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_14": {"frames": [0, 31, 62, 93, 124]},

    # task 015: tangerine large pink plate -> small teal plate
    "gigaworld0_gr1_sft/dreamgen_38_15": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_15": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_15": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_15": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_15": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_15": {"frames": [0, 31, 62, 93, 124]},

    # task 016: red tomato upper black tray -> brown paper bag
    "gigaworld0_gr1_sft/dreamgen_38_16": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_16": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_16": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_16": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_16": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_16": {"frames": [0, 31, 62, 93, 124]},

    # task 018: rubik's cube bottom wooden shelf -> top wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_18": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_18": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_18": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_18": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_18": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_18": {"frames": [0, 31, 62, 93, 124]},

    # task 023: grapes white table left side -> bottom level of shelf
    "gigaworld0_gr1_sft/dreamgen_38_23": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_23": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_23": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_23": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_23": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_23": {"frames": [0, 31, 62, 93, 124]},

    # task 029: yellow mustard bottle tan table -> middle white shelf
    "gigaworld0_gr1_sft/dreamgen_38_29": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_29": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_29": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_29": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_29": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_29": {"frames": [0, 31, 62, 93, 124]},

    # task 036: green apple top wooden shelf -> bottom wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_36": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_36": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_36": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_36": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_36": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_36": {"frames": [0, 31, 62, 93, 124]},

    # task 042: red pepper black top shelf -> bottom white shelf
    "gigaworld0_gr1_sft/dreamgen_38_42": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_42": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_42": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_42": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_42": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_42": {"frames": [0, 31, 62, 93, 124]},

    # task 048: yellow mango right side of table -> white shelf
    "gigaworld0_gr1_sft/dreamgen_38_48": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_48": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_48": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_48": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_48": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_48": {"frames": [0, 31, 62, 93, 124]},

    # task 049: green apple bottom shelf -> top shelf
    "gigaworld0_gr1_sft/dreamgen_38_49": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_49": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_49": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_49": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_49": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_49": {"frames": [0, 31, 62, 93, 124]},

    # task 052: ruik's cube bottom dark three-tier shelf -> top shelf
    "gigaworld0_gr1_sft/dreamgen_38_52": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_52": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_52": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_52": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_52": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_52": {"frames": [0, 31, 62, 93, 124]},

    # task 056: peach top black shelf -> brown paper bag
    "gigaworld0_gr1_sft/dreamgen_38_56": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_56": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_56": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_56": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_56": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_56": {"frames": [0, 31, 62, 93, 124]},

    # task 066: clock lower left shelf -> center of table
    "gigaworld0_gr1_sft/dreamgen_38_66": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_66": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_66": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_66": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_66": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_66": {"frames": [0, 31, 62, 93, 124]},

    # task 069: rubik's cube top three-tier shelf -> bottom brown shelf
    "gigaworld0_gr1_sft/dreamgen_38_69": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_69": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_69": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_69": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_69": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_69": {"frames": [0, 31, 62, 93, 124]},

    # task 079: rubik's cube top wooden shelf -> bottom wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_79": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_79": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_79": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_79": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_79": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_79": {"frames": [0, 31, 62, 93, 124]},

    # task 080: yellow mustard bottle tan table -> middle white shelf
    "gigaworld0_gr1_sft/dreamgen_38_80": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_80": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_80": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_80": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_80": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_80": {"frames": [0, 31, 62, 93, 124]},

    # task 086: clear orange cup top shelf -> bottom shelf
    # usable per annotation: gigaworld0_pretrain/dreamgen_58_86
    # usable per annotation: gigaworld0_pretrain/dreamgen_38_86; 38 never finished, but 58 finished first
    "gigaworld0_gr1_sft/dreamgen_38_86": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_86": {"frames": [0, 15, 22, 30, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_86": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_86": {"frames": [0, 7,10, 22, 39, 45]},
    "gigaworld0_pretrain/dreamgen_58_86": {"frames": [0, 10,15, 24, 30, 31]},
    "gigaworld0_pretrain/dreamgen_78_86": {"frames": [0, 31, 62, 93, 124]},

    # task 091: corn right side of table -> top two-tier wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_91": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_91": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_91": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_91": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_91": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_91": {"frames": [0, 31, 62, 93, 124]},
}

# Legacy tuple form kept too, so existing notes need no migration; prefer
# DREAMGEN_CASE_OVERRIDES for new edits.
DREAMGEN_FRAME_OVERRIDES = {
}

DREAMGEN_LAZINESS_TASKS = [
    ("1",
     "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
     "to_bottom_level_of_the_shelf",
     "Important: run generation; peach top->bottom shelf. Uniform fallback "
     "frames first, refine later."),
    ("4",
     "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
     "plate_to_center_of_teal_plate",
     "Important: purple glass blue plate->teal plate; fill in the 3.8/5.8/7.8s "
     "runs for pretrain/GR1-SFT."),
    ("5",
     "5_Use_the_right_hand_to_pick_up_green_apple_from_bottom_shelf_to_top_shelf",
     "Important: run generation; green apple bottom shelf->top shelf. Same "
     "3.8/5.8/7.8s frame logic as the others."),
    ("8",
     "8_Use_the_left_hand_to_pick_up_green_cucumber_from_from_the_beige_"
     "place_mat_to_to_the_small_cyan_plate",
     "Important: run generation; cucumber place mat->small cyan plate. "
     "Uniform fallback frames first."),
    ("14",
     "14_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_"
     "brown_wooden_shelf_to_top_level_of_brown_wooden_shelf",
     "Important: run generation; rubik's cube bottom->top level. Uniform "
     "fallback frames first."),
    ("15",
     "15_Use_the_right_hand_to_pick_up_the_tangerine_from_from_the_large_"
     "pink_plate_to_to_the_small_teal_plate",
     "Important: run generation; tangerine large pink plate->small teal plate. "
     "Uniform fallback frames first."),
    ("16",
     "16_Use_the_right_hand_to_pick_up_red_tomato_from_upper_black_tray_of_"
     "plastic_shelf_to_inside_of_brown_paper_bag",
     "Present, later disappeared; tomato upper black tray->paper bag. Uniform "
     "fallback frames first."),
    ("18",
     "18_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
     "wooden_shelf_to_top_level_of_the_wooden_shelf",
     "Present but harmless, keep it; rubik's cube bottom->top level. Uniform "
     "fallback frames first."),
    ("23",
     "23_Use_the_right_hand_to_pick_up_grapes_from_white_table_left_side_to_"
     "bottom_level_of_shelf",
     "Present but harmless, keep it; grapes table left side->bottom shelf. "
     "Uniform fallback frames first."),
    ("29",
     "29_Use_the_left_hand_to_pick_up_yellow_mustard_bottle_from_tan_table_"
     "to_middle_level_of_white_shelf",
     "Present but harmless, keep it; mustard bottle table->middle white shelf. "
     "Uniform fallback frames first."),
    ("36",
     "36_Use_the_right_hand_to_pick_up_green_apple_from_top_tier_of_wooden_"
     "shelf_to_bottom_tier_of_wooden_shelf",
     "Present but harmless, keep it; green apple top->bottom tier. Uniform "
     "fallback frames first."),
    ("42",
     "42_Use_the_right_hand_to_pick_up_red_pepper_from_black_top_shelf_to_"
     "bottom_white_shelf",
     "Important: run generation; red pepper black top shelf->white bottom "
     "shelf. Uniform fallback frames first."),
    ("48",
     "48_Use_the_right_hand_to_pick_up_yellow_mango_from_right_side_of_table_"
     "to_white_shelf",
     "Present but harmless, keep it; mango right side of table->white shelf. "
     "Uniform fallback frames first."),
    ("49",
     "49_Use_the_right_hand_to_pick_up_green_apple_from_bottom_shelf_to_top_"
     "shelf",
     "Present but harmless, keep it; green apple bottom->top shelf. Uniform "
     "fallback frames first."),
    ("52",
     "52_Use_the_right_hand_to_pick_up_ruik_s_cube_from_bottom_dark_wooden_"
     "three_tier_shelf_to_top_dark_wooden_three_tier_shelf",
     "Present but harmless, keep it; ruik_s_cube bottom->top tier. Uniform "
     "fallback frames first."),
    ("56",
     "56_Use_the_right_hand_to_pick_up_peach_from_top_black_shelf_to_inside_"
     "brown_paper_bag",
     "Important: run generation; peach black top shelf->paper bag. Uniform "
     "fallback frames first."),
    ("66",
     "66_Use_the_left_hand_to_pick_up_clock_from_lower_level_shelf_on_the_"
     "left_side_of_the_table_to_center_of_the_table",
     "No idea what to do next; clock lower left shelf->table center. Uniform "
     "fallback frames first."),
    ("69",
     "69_Use_the_right_hand_to_pick_up_rubik_s_cube_from_from_the_top_of_the_"
     "three_tiered_wooden_shelf_to_to_the_bottom_of_the_brown_three_tiered_"
     "wooden_shelf",
     "Rubik's cube top->bottom level; uniform fallback frames first."),
    ("79",
     "79_Use_the_right_hand_to_pick_up_rubik_s_cube_from_top_level_of_the_"
     "wooden_shelf_to_bottom_level_of_the_wooden_shelf",
     "Rubik's cube top->bottom level; uniform fallback frames first."),
    ("80",
     "80_Use_the_left_hand_to_pick_up_yellow_mustard_bottle_from_tan_table_"
     "to_middle_level_of_white_shelf",
     "Important: generate directly, make the other one disappear; mustard "
     "bottle table->middle white shelf."),
    ("86",
     "86_Use_the_right_hand_to_pick_up_clear_orange_cup_from_top_level_of_"
     "the_shelf_to_bottom_level_of_the_shelf",
     "Important: picks it back up; when refining later avoid selecting the "
     "human hand. Uniform fallback frames first."),
    ("91",
     "91_Use_the_right_hand_to_pick_up_corn_from_from_the_right_side_of_the_"
     "table_to_to_the_top_of_the_two_tiered_wooden_shelf",
     "No idea what to do next; corn right side of table->top of two-tier "
     "wooden shelf. Uniform fallback frames first."),
]


def dur_to_label(dur):
    """Convert internal dur code (58/98/158) to dir-friendly 5p8s/9p8s/15p8s."""
    s = str(dur)
    if len(s) >= 2 and s.isdigit():
        return f"{int(s[:-1])}p{s[-1]}s"
    return s


def task_to_dir(task):
    """Task number is the outermost dir; zero-pad numeric tasks for sorting."""
    s = str(task)
    return f"{int(s):03d}" if s.isdigit() else s


def _register_dreamgen_sweep_cases():
    added = 0
    for task_id, stem, note in DREAMGEN_LAZINESS_TASKS:
        for model, dur, run_dir in DREAMGEN_SWEEP_RUNS_3P8_5P8_7P8:
            video = os.path.join(DREAMGEN_SIDE_BY_SIDE_ROOT, run_dir, stem + ".mp4")
            if not os.path.isfile(video):
                continue
            key = f"{model}/dreamgen_{dur}_{task_id}"
            if key not in REGISTRY:
                added += 1
            entry = {
                "video": video,
                "frames": DREAMGEN_FRAME_OVERRIDES.get(
                    (model, dur, task_id),
                    DREAMGEN_DEFAULT_FRAMES_BY_DUR[dur],
                ),
                "note": f"{note} [{model}, {dur_to_label(dur)}]",
            }
            entry.update(DREAMGEN_CASE_OVERRIDES.get(key, {}))
            REGISTRY.setdefault(key, entry)
    return added


# Instruction source manifests (lazy-loaded on demand)
DREAMGEN_INPUT_JSON = (f"{GAGI_ROOT}/gr1_dreamgen_eval/"
                       "giga_input/gr1_dreamgen_it2v.json")
PBENCH_INPUT_JSON = f"{GAGIBENCH_ROOT}/pbench/giga_input/pbench_robot_it2v.json"

DEFAULT_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
REGISTERED_SWEEP_CASES = _register_dreamgen_sweep_cases()


# Video reading (cv2)
def open_video(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    return cap, w, h, n, fps


def read_right_half(cap, idx, w):
    """Read the right half (generated video) of frame idx; return an RGB ndarray."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx}")
    right = frame[:, w // 2:, :]          # columns [w//2 : w], full height
    return cv2.cvtColor(right, cv2.COLOR_BGR2RGB)


# Naming inference: model / bench / dur / task
def infer_bench(video_path):
    base = os.path.basename(video_path)
    low = video_path.lower()
    if base.startswith("robot_") or "pbench" in low:
        return "pbench"
    return "dreamgen"


def infer_model(video_path):
    low = video_path.lower()
    if "pretrain_dreamgen" in low or "pretrain" in low:
        return "gigaworld0_pretrain"
    if "gr1_dreamgen" in low or "sft_dreamgen" in low or "gr1_pbench" in low:
        return "gigaworld0_gr1_sft"
    if "sft" in low:
        return "gigaworld0_gr1_sft"
    return "gigaworld0"


def infer_dur(video_path, n_frames, fps):
    m = re.search(r"(\d+)p(\d+)s", video_path)
    if m:
        return f"{m.group(1)}{m.group(2)}"       # 9p8s -> 98
    # Fallback: estimate seconds from frames/fps, integer + first decimal
    sec = n_frames / (fps or 16.0)
    return f"{int(sec)}{int(round((sec - int(sec)) * 10))}"


def infer_task(video_path, bench):
    base = os.path.splitext(os.path.basename(video_path))[0]
    if bench == "pbench":
        m = re.search(r"robot_(\d+)", base)
        return str(int(m.group(1))) if m else base
    # dreamgen: leading N_ in the filename
    m = re.match(r"(\d+)_", base)
    return m.group(1) if m else base


# Instruction resolution
_dg_cache = None
_pb_cache = None


def resolve_instruction(video_path, bench):
    """Return (instruction_text, source_desc)."""
    global _dg_cache, _pb_cache
    base = os.path.basename(video_path)
    stem = os.path.splitext(base)[0]

    if bench == "dreamgen":
        # prefer the json lookup by request_id
        if _dg_cache is None and os.path.isfile(DREAMGEN_INPUT_JSON):
            try:
                _dg_cache = {d["request_id"]: d.get("prompt", "")
                             for d in json.load(open(DREAMGEN_INPUT_JSON))}
            except Exception:
                _dg_cache = {}
        if _dg_cache and stem in _dg_cache:
            return _dg_cache[stem], f"dreamgen_json[request_id={stem}]"
        # fallback: strip the N_ prefix, _ -> space
        txt = re.sub(r"^\d+_", "", stem).replace("_", " ").strip()
        return txt, "dreamgen_filename_fallback"

    # pbench: no instruction in the filename, look up json by pbench_id / index
    if _pb_cache is None and os.path.isfile(PBENCH_INPUT_JSON):
        try:
            data = json.load(open(PBENCH_INPUT_JSON))
            _pb_cache = {"by_id": {d.get("pbench_id"): d.get("prompt", "") for d in data},
                         "list": data}
        except Exception:
            _pb_cache = {"by_id": {}, "list": []}
    m = re.search(r"robot_(\d+)", stem)
    if _pb_cache and m:
        rid = f"robot_{int(m.group(1)):03d}"
        if rid in _pb_cache["by_id"]:
            return _pb_cache["by_id"][rid], f"pbench_json[pbench_id={rid}]"
        idx = int(m.group(1))
        if 0 <= idx < len(_pb_cache["list"]):
            return _pb_cache["list"][idx].get("prompt", ""), f"pbench_json[index={idx}]"
    return stem, "pbench_stem_fallback"


# Right-half video export (official crop convention): prefer imageio_ffmpeg, fall back to
# cv2 VideoWriter
def export_right_video(src, dst, w, h, fps):
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-i", src, "-vf", "crop=iw/2:ih:iw/2:0",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "18", dst]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        if r.returncode == 0 and os.path.isfile(dst):
            return "imageio_ffmpeg crop=iw/2:ih:iw/2:0"
        sys.stderr.write(f"[warn] ffmpeg failed, falling back to cv2: {r.stderr.decode()[:200]}\n")
    except Exception as e:
        sys.stderr.write(f"[warn] imageio_ffmpeg unavailable ({e}), "
                         f"falling back to cv2 VideoWriter\n")

    # Fallback: re-encode frame by frame with the right half cropped by cv2
    cap = cv2.VideoCapture(src)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(dst, fourcc, fps, (w - w // 2, h))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr[:, w // 2:, :])
    cap.release()
    vw.release()
    return "cv2_VideoWriter fallback"


# Montage
def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_montage(frames_rgb, labels, out_path, caption="", sep=4,
                 caption_h=0, label_h=26):
    """Equal-height horizontal concat. labels caption each frame below (e.g. 'f2 idx078 4.9s');
    caption is the instruction on top."""
    imgs = [Image.fromarray(f) for f in frames_rgb]
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    cap_h = caption_h if caption else 0
    total_w = sum(im.width for im in imgs) + sep * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, cap_h + h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if caption:
        draw.text((6, 4), caption, fill=(0, 0, 0), font=_font(18))
    x = 0
    for im, lab in zip(imgs, labels):
        canvas.paste(im, (x, cap_h))
        if lab:
            draw.text((x + 4, cap_h + h + 3), lab, fill=(0, 0, 0), font=_font(15))
        x += im.width + sep
    canvas.save(out_path)
    return canvas.size


# Mode implementations
def run_survey(video, n, out_root):
    cap, w, h, nf, fps = open_video(video)
    bench = infer_bench(video)
    model = infer_model(video)
    dur = infer_dur(video, nf, fps)
    task = infer_task(video, bench)
    idxs = [int(round(i)) for i in np.linspace(0, nf - 1, n)]
    frames = [read_right_half(cap, i, w) for i in idxs]
    cap.release()
    labels = [f"idx{ix:03d} {ix/fps:.2f}s" for ix in idxs]
    instr, src = resolve_instruction(video, bench)
    d = os.path.join(out_root, "survey", task_to_dir(task), model,
                     f"{bench}_{dur_to_label(dur)}")
    os.makedirs(d, exist_ok=True)
    cap_txt = (f"[SURVEY {n} frames] task={task} {model} "
               f"{bench}_{dur_to_label(dur)} | {instr}")
    make_montage(frames, labels, os.path.join(d, "survey.png"),
                 caption=cap_txt, caption_h=26)
    print(f"[survey] {nf} frames @ {fps:.1f} fps -> {os.path.join(d, 'survey.png')}")
    print(f"[survey] instruction ({src}): {instr}")
    print(f"[survey] uniform frame indices: {idxs}")
    print(f"[survey] inspect this strip, pick 4-6 laziness frames, then try "
          f"--video ... --frames a,b,c,d.")
    return d


def build_case(video, frames, out_root, model_override=None,
               bench_override=None, note="", no_caption=False, case_id=None,
               print_snippet=True):
    cap, w, h, nf, fps = open_video(video)
    bench = bench_override or infer_bench(video)
    model = model_override or infer_model(video)
    dur = infer_dur(video, nf, fps)
    task = infer_task(video, bench)

    if not (4 <= len(frames) <= 6):
        raise ValueError(f"frames must be 4-6, got {len(frames)}: {frames}")
    for ix in frames:
        if not (0 <= ix < nf):
            raise ValueError(f"frame index {ix} out of range [0,{nf-1}]")

    out_dir = os.path.join(out_root, task_to_dir(task), model,
                           f"{bench}_{dur_to_label(dur)}")
    os.makedirs(out_dir, exist_ok=True)

    # extract and save frames
    imgs = []
    for k, ix in enumerate(frames):
        rgb = read_right_half(cap, ix, w)
        imgs.append(rgb)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"f{k}_idx{ix:03d}.png"))
    cap.release()

    instr, src = resolve_instruction(video, bench)
    # montage
    labels = [f"f{k} idx{ix:03d} {ix/fps:.2f}s" for k, ix in enumerate(frames)]
    caption = ("" if no_caption else
               f"task={task} {model} {bench}_{dur_to_label(dur)} | {instr}")
    make_montage(imgs, labels, os.path.join(out_dir, "montage.png"),
                 caption=caption, caption_h=26 if caption else 0)

    # right-half video
    right_mode = export_right_video(video, os.path.join(out_dir, "right_video.mp4"),
                                    w, h, fps)

    # instruction metadata
    meta = {
        "case_id": case_id or f"{model}/{bench}_{dur}_{task}",
        "instruction": instr, "instruction_source": src,
        "source_video": video, "model": model, "bench": bench,
        "dur": dur, "task": task, "fps": fps, "num_frames": nf,
        "video_wh": [w, h], "right_half_x": w // 2,
        "selected_frames": list(frames),
        "selected_times_sec": [round(ix / fps, 3) for ix in frames],
        "output_layout": "<out_root>/<task>/<model>/<bench>_<dur>",
        "note": note, "right_video_mode": right_mode,
    }
    with open(os.path.join(out_dir, "instruction.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "instruction.txt"), "w") as f:
        f.write(instr + "\n")

    print(f"[case] -> {out_dir}")
    print(f"[case] instruction ({src}): {instr}")
    print(f"[case] frames {frames}  seconds {meta['selected_times_sec']}")
    print(f"[case] right_video: {right_mode}")
    # print a paste-ready registry snippet (try until satisfied, then keep it)
    if print_snippet:
        print("\n# ---- Paste this into REGISTRY: ----")
        print(f'    "{model}/{bench}_{dur}_{task}": {{')
        print(f'        "video": "{video}",')
        print(f'        "frames": {list(frames)},')
        print(f'        "note": "{note}",')
        print(f'    }},')
    return out_dir


def parse_frames(s):
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def select_registry_keys(id_prefix=""):
    prefixes = [p.strip() for p in str(id_prefix).split(",") if p.strip()]
    keys = sorted(REGISTRY, key=_registry_sort_key)
    if not prefixes:
        return keys
    return [k for k in keys if any(k.startswith(p) for p in prefixes)]


def _registry_sort_key(key):
    model, rest = key.split("/", 1) if "/" in key else ("", key)
    m = re.match(r"([a-zA-Z0-9]+)_(\d+)_(\d+)$", rest)
    if m:
        bench, dur, task = m.groups()
        task_key = int(task) if task.isdigit() else task
        dur_key = int(dur) if dur.isdigit() else dur
        return (task_key, model, bench, dur_key)
    return (999999, model, rest, 999999)


def run_registry_id(case_id, args, print_snippet=True):
    e = REGISTRY[case_id]
    model = args.model or (case_id.split("/")[0] if "/" in case_id else "")
    return build_case(e["video"], e["frames"], args.out_root,
                      model_override=model or None,
                      bench_override=args.bench or None,
                      note=e.get("note", ""), no_caption=args.no_caption,
                      case_id=case_id, print_snippet=print_snippet)


def main():
    ap = argparse.ArgumentParser(description="GigaWorld-0 laziness frame extraction + montage")
    ap.add_argument("--survey", type=int, default=0,
                    help="survey mode: uniformly sample N frames (e.g. 16)")
    ap.add_argument("--id", type=str, default="",
                    help="id mode: REGISTRY key, e.g. gigaworld0/dreamgen_98_1")
    ap.add_argument("--all", action="store_true",
                    help="run the whole REGISTRY; pair with --id-prefix to limit prefixes")
    ap.add_argument("--list", action="store_true",
                    help="list REGISTRY keys; pair with --id-prefix to limit prefixes")
    ap.add_argument("--id-prefix", type=str, default="",
                    help="comma-separated key prefixes, e.g. "
                         "gigaworld0_pretrain/,gigaworld0_gr1_sft/")
    ap.add_argument("--video", type=str, default="", help="video mode: mp4 path")
    ap.add_argument("--frames", type=str, default="",
                    help="comma-separated frame indices, e.g. 0,44,78,100")
    ap.add_argument("--times", type=str, default="",
                    help="comma-separated seconds (converted via fps)")
    ap.add_argument("--model", type=str, default="",
                    help="override the model name (cross-model stages)")
    ap.add_argument("--bench", type=str, default="", choices=["", "dreamgen", "pbench"])
    ap.add_argument("--note", type=str, default="", help="laziness description")
    ap.add_argument("--no-caption", action="store_true",
                    help="omit the instruction caption from the montage")
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    a = ap.parse_args()

    if a.survey > 0:
        if not a.video:
            ap.error("--survey requires --video")
        run_survey(a.video, a.survey, a.out_root)
        return 0

    if a.list:
        keys = select_registry_keys(a.id_prefix)
        print(f"[list] {len(keys)} cases")
        for k in keys:
            print(k)
        return 0

    if a.all:
        keys = select_registry_keys(a.id_prefix)
        if not keys:
            ap.error(f"--all matched no REGISTRY key: {a.id_prefix}")
        print(f"[all] running {len(keys)} cases")
        for i, key in enumerate(keys, 1):
            print(f"\n[all] {i}/{len(keys)} {key}")
            run_registry_id(key, a, print_snippet=False)
        print(f"\n[all] done: {len(keys)} cases -> {a.out_root}")
        return 0

    if a.id:
        if a.id not in REGISTRY:
            ap.error(f"no such REGISTRY id: {a.id}; have: {list(REGISTRY)}")
        if not REGISTRY[a.id].get("frames"):
            ap.error(f"frames empty for {a.id}; survey first and fill them into REGISTRY")
        run_registry_id(a.id, a)
        return 0

    if a.video:
        frames = None
        if a.frames:
            frames = parse_frames(a.frames)
        elif a.times:
            _, w, h, nf, fps = open_video(a.video)
            frames = [int(round(float(t) * fps)) for t in a.times.split(",")]
        if not frames:
            ap.error("--video needs --frames or --times")
        build_case(a.video, frames, a.out_root,
                   model_override=a.model or None, bench_override=a.bench or None,
                   note=a.note, no_caption=a.no_caption)
        return 0

    ap.error("specify --survey N | --id KEY | --video PATH (--frames/--times)")


if __name__ == "__main__":
    sys.exit(main())
