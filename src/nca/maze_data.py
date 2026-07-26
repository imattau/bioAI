import torch

from src.vsa import VSA


def make_checkerboard(grid_size: int = 28, cell_size: int = 4) -> torch.Tensor:
    grid = torch.zeros(1, 1, grid_size, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            cell_i = i // cell_size
            cell_j = j // cell_size
            if (cell_i + cell_j) % 2 == 0:
                grid[0, 0, i, j] = 1.0
    return grid


def make_stripes(grid_size: int = 28, n_stripes: int = 4) -> torch.Tensor:
    grid = torch.zeros(1, 1, grid_size, grid_size)
    stripe_w = grid_size // n_stripes
    for s in range(n_stripes):
        if s % 2 == 0:
            grid[0, 0, :, s * stripe_w : (s + 1) * stripe_w] = 1.0
    return grid


def make_plus(grid_size: int = 28, arm_w: int = 6) -> torch.Tensor:
    grid = torch.zeros(1, 1, grid_size, grid_size)
    center = grid_size // 2
    half_w = arm_w // 2
    grid[0, 0, :, center - half_w : center + half_w + 1] = 1.0
    grid[0, 0, center - half_w : center + half_w + 1, :] = 1.0
    return grid


def make_box(grid_size: int = 28, margin: int = 4) -> torch.Tensor:
    grid = torch.zeros(1, 1, grid_size, grid_size)
    grid[0, 0, margin:-margin, margin:-margin] = 1.0
    return grid


def make_hollow_box(grid_size: int = 28, margin: int = 4,
                    wall_w: int = 2) -> torch.Tensor:
    grid = torch.zeros(1, 1, grid_size, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            near_top = i < margin + wall_w and i >= margin
            near_bot = i >= grid_size - margin - wall_w and i < grid_size - margin
            near_left = j < margin + wall_w and j >= margin
            near_right = j >= grid_size - margin - wall_w and j < grid_size - margin
            vert_wall = (i >= margin and i < grid_size - margin) and (near_left or near_right)
            horiz_wall = (j >= margin and j < grid_size - margin) and (near_top or near_bot)
            if vert_wall or horiz_wall:
                grid[0, 0, i, j] = 1.0
    return grid


PATTERN_MAKERS = {
    "checkerboard": make_checkerboard,
    "stripes": make_stripes,
    "plus": make_plus,
    "box": make_box,
    "hollow_box": make_hollow_box,
}


def build_dataset(vsa: VSA, grid_size: int = 28,
                  pattern_names: list[str] | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if pattern_names is None:
        pattern_names = list(PATTERN_MAKERS.keys())

    dataset = []
    for name in pattern_names:
        maker = PATTERN_MAKERS[name]
        target = maker(grid_size=grid_size)
        vsa_vector = vsa.make_vector()
        dataset.append((vsa_vector, target))
    return dataset
