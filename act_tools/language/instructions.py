"""Instruction sets for the language-conditioned wiping task.

TRAIN paraphrases are used in the generated demonstrations; HELD_OUT paraphrases never appear in
training and test whether the policy follows unseen wording. "left"/"right" are the two drawn
regions as seen by the robot (left region at y = -0.09 m, right at y = -0.27 m).
"""
TRAIN = {
    'left': ['wipe the left area', 'clean the left side', 'wipe the area on the left',
             'scrub the left region', 'please wipe the left patch'],
    'right': ['wipe the right area', 'clean the right side', 'wipe the area on the right',
              'scrub the right region', 'please wipe the right patch'],
}
HELD_OUT = {
    'left': ['clean up the left part of the table', 'the left square needs wiping',
             'use the sponge on the left one'],
    'right': ['clean up the right part of the table', 'the right square needs wiping',
              'use the sponge on the right one'],
}
