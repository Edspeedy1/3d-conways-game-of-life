import numpy as np
from stl import mesh
import pygame
import time

GridSize = 30
Expand = 10
hieght = 100
baseExtra = 10

# Function to apply Conway's Game of Life rules to 3D grid
def apply_rules(grid):
    # Count neighbors for each cell
    neighbor_count = np.zeros_like(grid, dtype=int)
    rows, cols = grid.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            # Roll the grid and account for edge conditions
            rolled_grid = np.roll(np.roll(grid, i, axis=0), j, axis=1)
            if i == -1:
                rolled_grid[-1, :] = 0  # Set bottom row to 0
            elif i == 1:
                rolled_grid[0, :] = 0   # Set top row to 0
            if j == -1:
                rolled_grid[:, -1] = 0  # Set rightmost column to 0
            elif j == 1:
                rolled_grid[:, 0] = 0   # Set leftmost column to 0
            neighbor_count += rolled_grid
    
    # Apply rules
    new_grid = np.logical_and(neighbor_count == 3, np.logical_not(grid))
    new_grid |= np.logical_and(np.logical_or(neighbor_count == 2, neighbor_count == 3), grid)
    
    return new_grid

def fill_enclosed_zeros(matrix):

    # matrix to python int matrix
    matrix = [[int(cell) for cell in row] for row in matrix]
    

    rows = len(matrix)
    cols = len(matrix[0])

    stack = [(i,j) for i in range(rows) for j in range(cols) if matrix[i][j] == 0 and (i in [0, rows - 1] or j in [0, cols - 1])]

    while stack:
        i, j = stack.pop()
        matrix[i][j] = 2  # Temporary mark for visited
        if i + 1 < rows and matrix[i + 1][j] == 0:
            stack.append((i + 1, j))
        if i - 1 >= 0 and matrix[i - 1][j] == 0:
            stack.append((i - 1, j))
        if j + 1 < cols and matrix[i][j + 1] == 0:
            stack.append((i, j + 1))
        if j - 1 >= 0 and matrix[i][j - 1] == 0:
            stack.append((i, j - 1))

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 2:
                matrix[i][j] = 0
            elif matrix[i][j] == 0:
                matrix[i][j] = 1

    return np.array(matrix)

def expand_grid(grid, size):
    expanded_grid = np.zeros((grid.shape[0] + size*2, grid.shape[1] + size*2), dtype=grid.dtype)
    expanded_grid[size:size+grid.shape[0], size:size+grid.shape[1]] = grid
    return expanded_grid

# Function to export 3D grid to STL file
def array_to_stl(array, filename):
    # Vertices of a unit cube centered at the origin
    vertices = np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1],[1, 1, 1],[0, 1, 1]])

    # Define the eight vertices of the cube
    faces = np.array([[0, 1, 2],[0, 2, 3],[3, 2, 6],[3, 6, 7],[7, 6, 5],[7, 5, 4],[4, 5, 1],[4, 1, 0],[2, 1, 5],[2, 5, 6],[4, 0, 3],[4, 3, 7]])

    # Create a new mesh
    total_faces = len(faces)
    total_cells = sum(cell.sum() for cell in array)
    cube_mesh = mesh.Mesh(np.zeros(total_faces * total_cells, dtype=mesh.Mesh.dtype))

    # Rotation matrix around the y-axis by 90 degrees
    rotation_matrix = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]])

    # Fill the mesh with vertices and faces based on array
    idx = 0
    for i, cell in enumerate(array):
        for j, row in enumerate(cell):
            for k, val in enumerate(row):
                if val == 1:
                    cube_offset = np.array([i, j, k])
                    if i == 0:
                        cube_offset = np.array([i, j - baseExtra/2 , k - baseExtra/2 ])
                    rotated_vertices = vertices + cube_offset
                    rotated_vertices = np.dot(rotated_vertices, rotation_matrix)
                    for f in faces:
                        for l in range(3):
                            cube_mesh.vectors[idx][l] = rotated_vertices[f[l]]
                        idx += 1

    # Write the mesh to file "filename"
    cube_mesh.save(filename)

def draw_array(array, window):
    window.fill((0, 0, 0))
    for i, cell in enumerate(array):
        for j, val in enumerate(cell):
            if val == 1:
                pygame.draw.rect(window, (255, 255, 255), (i * 10, j * 10, 10, 10))

    pygame.display.update()


# Example usage
if __name__ == "__main__":
    window = pygame.display.set_mode((800, 800))
    # Set up initial 3D grid
    grid3D = [expand_grid(np.random.choice([1, 1], size=(GridSize+baseExtra, GridSize+baseExtra), p=[0.5, 0.5]), Expand)] # expand_grid(np.random.choice([1, 1], size=(GridSize+5, GridSize+5), p=[0.5, 0.5]), Expand)
    grid = np.random.choice([1, 0], size=(GridSize, GridSize), p=[0.5, 0.5])
    grid = expand_grid(grid, Expand)
    # grid3D.append(grid)

    # Run simulation for some iterations
    for _ in range(hieght):
        grid = apply_rules(grid)
        draw_array(grid, window)
        grid_copy = grid.copy()  # Create a copy of the grid
        grid3D.append(fill_enclosed_zeros(grid_copy))
    
    # Export to STL
    array_to_stl(grid3D, f"outputs\\game_of_life {time.time()}.stl")

