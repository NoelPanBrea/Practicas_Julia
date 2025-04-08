import pygame
import time
from julia import Main
from _collections_abc import Generator
#Main.include("73166321D_54157616E_48118254T_54152126Y.jl")
Main.include("Ej2.jl")
model = Main.ann
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
        self.load_path = "optical+recognition+of+handwritten+digits\small.txt"
        self.save_path = "optical+recognition+of+handwritten+digits\generated.txt"
        self.loaded = self.readfile()

    def launch(self) -> None:
        fmenu = False
        self.cell_init()
        self.clean_button = pygame.Rect(self.size[0] - 100, 0, 100, self.size[1]/4)
        self.load_button = pygame.Rect(self.size[0] - 100, self.size[1]/4, 100, self.size[1]/4)
        self.save_button = pygame.Rect(self.size[0] - 100, 2 * self.size[1]/4, 100, self.size[1]/4)
        self.predict_button = pygame.Rect(self.size[0] - 100, 3 * self.size[1]/4, 100, self.size[1]/4)
        self.brush = pygame.Rect(0, 0, self.thickness, self.thickness)
        cooldown = 5
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
                if time.time() - cooldown > 1:
                    self.load()
                    cooldown = time.time()
            elif pressed[0] and self.save_button.collidepoint(pos):
                if time.time() - cooldown > 1:
                    self.save()
                    cooldown = time.time()
            elif pressed[0] and self.predict_button.collidepoint(pos):
                if time.time() - cooldown > 1:
                    print("respuesta", self.predict().tolist().index(True))
                    cooldown = time.time()

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
        pygame.draw.rect(self.win, (255, 255, 0), self.predict_button)
        for row in self.cells:
            for cell in row:
                pygame.draw.rect(self.win, cell[1], cell[0])
        pygame.display.update()

    def readfile(self) -> Generator:
        with open(self.load_path, "r") as file:
            array = file.read().replace("\n", ",").split(",")[:-1]
        for i in range(len(array)):
            yield array[i*65:65 * (i + 1)]

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
        with open(self.save_path, "a") as file:
            for i in range(64):
                cellx = 4 * (i % 8)
                celly = 4 * (i // 8)
                coeff = 0
                for j in range(4):
                    for k in range(4): 
                            coeff += 1 if self.cells[celly + j][cellx + k][1] == (0, 0, 0) else 0
                file.write(str(coeff) + ",")
            file.write("0,")
                

    def load(self):
        try:
            number = next(self.loaded)
            print(number[-1])
            for i, x in enumerate(number[:-1]):
                cellx = 4 * (i % 8)
                celly = 4 * (i // 8)
                if int(x) > 0:
                    for j in range(4):
                        for k in range(4): 
                            self.cells[celly + j][cellx + k][1] = (0, 0, 0)
        except Exception as e:
            pass

    def clean(self):
        for row in self.cells:
            for cell in row:
                cell[1] = (255, 255, 255)

    def predict(self):
        array = []
        for i in range(64):
            cellx = 4 * (i % 8)
            celly = 4 * (i // 8)
            coeff = 0
            for j in range(4):
                for k in range(4): 
                        coeff += 1 if self.cells[celly + j][cellx + k][1] == (0, 0, 0) else 0
            array.append(coeff)
        return Main.classifyOutputs(model(array))
                

            

def main() -> None:
    big = (839, 739)
    small = (519, 419)
    main_window = Window(big, 32, 255)
    #main_window.launch()


if __name__ == "__main__":
    main()