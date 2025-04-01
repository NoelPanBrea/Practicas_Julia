import pygame

class Window:
    def __init__(self, size: tuple, gridsize: int, fps: int) -> None:
        pygame.init()
        self.size = size
        self.win = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.bgcolor = (0, 0, 0)
        self.gridsize = gridsize
        self.borderpx = 3
        self.thickness = 40
        self.load_path = ""
        self.save_path = ""

    def launch(self) -> None:
        fmenu = False
        self.cell_init()
        self.clean_button = pygame.Rect(self.size[0] - 100, 0, 100, self.size[1]/3)
        self.load_button = pygame.Rect(self.size[0] - 100, self.size[1]/3, 100, self.size[1]/3)
        self.save_button = pygame.Rect(self.size[0] - 100, 2 * self.size[1]/3, 100, self.size[1]/3)
        self.brush = pygame.Rect(0, 0, self.thickness, self.thickness)
        while not fmenu:
            self.clock.tick(self.fps)
            pos = pygame.mouse.get_pos()
            self.brush.x, self.brush.y = pos
            pressed = pygame.mouse.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            if pressed[0] and self.clean_button.collidepoint(pos):
                self.clean()
            elif pressed[0] and self.load_button.collidepoint(pos):
                self.load()
            elif pressed[0] and self.save_button.collidepoint(pos):
                self.save()
            for row in self.cells:
                for cell in row:
                    cell[1] = (0, 0, 0) if cell[0].colliderect(self.brush) and pressed[0] else cell[1]
            self.draw()


    def draw(self) -> None:
        self.win.fill(self.bgcolor)
        pygame.draw.rect(self.win, (255, 255, 255), [self.size[0] - 100, 0, 100, self.size[1]])
        pygame.draw.rect(self.win, (255, 0, 0), self.clean_button)
        pygame.draw.rect(self.win, (0, 0, 255), self.load_button)
        pygame.draw.rect(self.win, (0, 255, 0), self.save_button)
        for row in self.cells:
            for cell in row:
                pygame.draw.rect(self.win, cell[1], cell[0])
        pygame.display.update()


    def cell_init(self) -> list[list[pygame.Rect]]:
        tablesize = (self.size[0] - 100, self.size[1])
        cellsize = (tablesize[0] - (self.borderpx * (self.gridsize + 1))) / self.gridsize
        color = (255, 255, 255)
        self.cells = []
        for i in range(self.gridsize):
            row = []
            for j in range(self.gridsize):
                rect = pygame.Rect(self.borderpx + j * (cellsize + self.borderpx), self.borderpx + i * (cellsize + self.borderpx), cellsize, cellsize)
                row.append([rect, color])
            self.cells.append(row)

    def save(self):
        for row in self.cells:
            for cell in row:
                pass

    def load(self):
        pass
    
    def clean(self):
        for row in self.cells:
            for cell in row:
                cell[1] = (255, 255, 255)
                

            

def main() -> None:
    big = (839, 739)
    small = (519, 419)
    main_window = Window(big, 32, 255)
    main_window.launch()


if __name__ == "__main__":
    main()