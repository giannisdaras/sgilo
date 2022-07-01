from ml_collections import config_dict
from collections import OrderedDict
from manim import *

import os
from PIL import Image
import numpy as np
import itertools
from derivative_proxy import g, g_proxy
cfg = config_dict.ConfigDict()
cfg.background_color = "#ffffff"
# cfg.background_color = "#ece6e2"  # white
cfg.text_color = "#000000"  # black
cfg.image_resolution = 2048

cfg.text_scale = 0.6

# corners
cfg.left = np.array([-7, 0, 0])
cfg.right = np.array([7, 0, 0])
cfg.down = np.array([0, -4, 0])
cfg.up = np.array([0, 4, 0])


cfg.inp_resolution = 1300


od = OrderedDict()
od["Giannis Daras (*)"] = "daras.png"
od["Yuval Dagan (*)"] = "yuval.png"
od["Alexandros Dimakis"] = "dimakis.png"
od["Constantinos Daskalakis"] = "costis.png"


def connect_shapes(obj1, obj2, color=WHITE):
    arrow_up = Arrow(start=obj1.get_corner(RIGHT + UP), 
        end=obj2.get_corner(LEFT + UP), color=color)
    arrow_down = Arrow(start=obj1.get_corner(RIGHT + DOWN), 
        end=obj2.get_corner(LEFT + DOWN), color=color)
    return arrow_up, arrow_down


def central_crop(image_path, width, height):
    """
    Crop the image to the center of the image.
    :param image_path: path to the image
    :param width: width of the image
    :param height: height of the image
    :return: cropped image
    """
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image = image.crop(((width - height) // 2, 0, (width + height) // 2, height))
    return image


class Person:
    def __init__(self, name, image):
        self.name = Tex(name, color=cfg.text_color).scale(cfg.text_scale)
        self.image = ImageMobject(central_crop(image, 256, 256))
        self.name.next_to(self.image, UP)
        self.group = Group(self.image, self.name)
    
    def __str__(self):
        return self.name


class Intro(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        main_text = Tex(r"Score-Guided Intermediate Layer Optimization: \\ Fast Langevin Mixing for Inverse Problems", color=cfg.text_color)
        main_text.shift(UP * 2)
        self.add(main_text)

        persons = [Person(name, os.path.join("images/authors", image)).group for name, image in od.items()]
        persons_group = Group(*persons).arrange()
        persons_group.next_to(main_text, 2.0 * DOWN)
        self.add(persons_group)


        equal_contrib = Tex(r"(*): equal contribution", color=cfg.text_color).scale(cfg.text_scale)
        # equal_contrib.next_to(cfg.left + cfg.down, RIGHT)
        equal_contrib.align_on_border(LEFT + DOWN)
        self.add(equal_contrib)
        # self.play()
        self.wait()
        # self.endSlide()

class Teaser(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        teaser_text = Tex(r"Contributions of our work", color=cfg.text_color)
        teaser_text.next_to(cfg.up, DOWN)
        self.add(teaser_text)

        vertical_line = Line(cfg.down, teaser_text.get_center() + 0.3 * DOWN, color=cfg.text_color)
        self.add(vertical_line)

        horizontal_line = Line(cfg.left, cfg.right, color=cfg.text_color)
        self.add(horizontal_line)
        
        contrib1 = Tex(r"$\rhd$ Solve inverse problems with extremely \\ sparse measurements", color=cfg.text_color).scale(cfg.text_scale)
        contrib1.next_to(teaser_text, DOWN).align_on_border(LEFT)
        self.add(contrib1)

        inp = ImageMobject("images/paper_figs/inpainting_input.png", scale_to_resolution=cfg.inp_resolution)
        sgilo_inp = ImageMobject("images/paper_figs/inpainting_sgilo.png", scale_to_resolution=cfg.inp_resolution)
        reference = ImageMobject("images/paper_figs/inpainting_ref.png", scale_to_resolution=cfg.inp_resolution)
        contrib1_graphic = Group(inp, sgilo_inp, reference).arrange(RIGHT, center=True)
        self.add(contrib1_graphic)
        self.add(reference)
        
        self.wait()
        self.add(inp)
        self.wait()
        self.add(sgilo_inp)
        self.wait()
        self.play(contrib1_graphic.animate.scale(0.7).move_to(contrib1.get_bottom() + 1.3 * DOWN))

        contrib2 = Tex(r"$\rhd$ Convergence of Langevin for Inverse Problems \\ with generative models", color=cfg.text_color).scale(cfg.text_scale)
        contrib2.next_to(vertical_line, RIGHT).set_y(contrib1.get_y())
        self.add(contrib2)
        contrib2_graphic = ImageMobject("images/vector_field.png", scale_to_resolution=1.2 * cfg.inp_resolution)
        self.add(contrib2_graphic)
        self.wait()
        self.play(contrib2_graphic.animate.scale(0.5).move_to(contrib2.get_bottom() + 1.3 * DOWN))

        contrib3 = Tex(r"$\rhd$ Posterior sampling with GANs", color=cfg.text_color).scale(cfg.text_scale)
        contrib3.next_to(vertical_line, LEFT).align_on_border(LEFT)
        self.add(contrib3)
        contrib3_graphic = ImageMobject("images/paper_figs/gan_posterior.png", scale_to_resolution=cfg.inp_resolution)
        self.add(contrib3_graphic)
        self.wait()
        self.play(contrib3_graphic.animate.scale(0.5).move_to(contrib3.get_bottom() + DOWN))

        contrib4 = Tex(r"$\rhd$ GANs + Diffusion = $\heartsuit$ for inverse problems", color=cfg.text_color).scale(cfg.text_scale)
        contrib4.next_to(vertical_line, RIGHT).set_y(contrib3.get_y())
        self.add(contrib4)

        frogs = ImageMobject("images/paper_figs/frogs.png", scale_to_resolution=1.2 * cfg.inp_resolution)
        mona_lisa = ImageMobject("images/paper_figs/mona_lisa.png", scale_to_resolution=0.45 * cfg.inp_resolution)
        contrib4_graphic = Group(frogs, mona_lisa).scale(0.5).arrange(DOWN, center=True)
        self.add(contrib4_graphic)
        self.wait()
        self.play(contrib4_graphic.animate.scale(0.5).move_to(contrib4.get_bottom() + 1.2 * DOWN))
        self.wait()

# generator 
gen_width = 0.5
num_layers = 4
layers_dist = 6
gen_color = PURPLE
loss_width = 4.5
loss_height = 1.5
loss_color = RED
image_resolution = 2048

class CSGM(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        teaser_text = Tex(r"Prior work: CSGM (ICML 2017)", color=cfg.text_color)
        teaser_text.next_to(cfg.up, DOWN)
        self.add(teaser_text)
        self.wait()
        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index)).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.8).shift(0.5 * DOWN)
        self.play(FadeIn(generator))
        self.wait()
        # self.endSlide()

        loss = MathTex(r"\left|\left|G_3G_2G_1(z_0) - x\right|\right|^2", color=cfg.text_color)
        loss_rec = Rectangle(height=loss_height, width=loss_width, color=loss_color)
        loss_rec.next_to(0.5 * layers[1].get_center() + 
            0.5 * layers[2].get_center(), 12 * DOWN + 0.25 * RIGHT)
        loss.move_to(loss_rec.get_center())


        # smooth connections
        dot1 = Dot(color=cfg.text_color)
        dot1.next_to(layers[-1], 10 * RIGHT)
        x_text = MathTex(r"x", color=cfg.text_color)
        x_text.next_to(dot1, UP)

        x_image = ImageMobject("images/authors/dimakis.png", scale_to_resolution=1.00 * image_resolution)
        x_image.next_to(x_text, RIGHT)

        dot2 = Dot(color=cfg.text_color)
        dot2.next_to(dot1, 12 * DOWN)
        dot2.set_y(loss_rec.get_center()[1])

        dot3 = Dot(color=cfg.text_color)
        dot3 = dot3.next_to(dot2, LEFT)
        rec_right_x = (0.5 * loss_rec.get_corner(RIGHT + UP) + 0.5 * loss_rec.get_corner(DOWN + RIGHT))[0]
        dot3.set_x(rec_right_x)



        n_arrows = []
        things_to_connect = [layers[-1].get_center(), dot1, dot2, dot3]
        for index in range(0, len(things_to_connect) - 1):
            n_arrows.append(Arrow(start=things_to_connect[index], 
                end=things_to_connect[index + 1], color=cfg.text_color))
        
        dot4 = Dot(color=cfg.text_color)
        dot4.next_to(loss_rec, LEFT)
        rec_left_x = (0.5 * loss_rec.get_corner(LEFT + UP) + 0.5 * loss_rec.get_corner(DOWN + LEFT))[0]
        dot4.set_x(rec_left_x)
        
        dot5 = Dot(color=cfg.text_color)
        dot5.next_to(dot4, LEFT)
        dot5.set_x(first_text.get_center()[0])

        n_arrows.append(Arrow(start=dot4, end=dot5, color=cfg.text_color))
        n_arrows.append(Arrow(start=dot5, end=first_text, color=cfg.text_color))


        scene_components = [loss, loss_rec, dot1, dot2, dot3, dot4, dot5, *n_arrows, x_text, x_image]
        self.play(*[FadeIn(x) for x in scene_components])
        self.wait()
        # self.endSlide()

        mobj = Group(generator, *scene_components)
        orig_coords = mobj.get_center()
        factor = 0.5
        new_width = mobj.width * factor
        new_height = mobj.height * factor
        self.play(
            mobj.animate.stretch_to_fit_height(
                new_height).stretch_to_fit_width(
                    new_width).align_on_border(UP + LEFT).shift(DOWN))
        self.wait()
        # self.endSlide()

        rec_text = MathTex(r"G(z_0*)", color=gen_color)
        rec_image = ImageMobject("images/alex_csgm.png", scale_to_resolution=0.5 * image_resolution)
        rec_image.move_to(np.array([3., 0., 0.]))
        rec_text.next_to(rec_image, DOWN)
        self.play(FadeIn(rec_image), FadeIn(rec_text))
        self.wait()
        # self.endSlide()

        self.play(FadeOut(rec_text), FadeOut(rec_image), FadeOut(mobj))
        self.wait()
        # self.endSlide()


        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.8).shift(0.5 * DOWN)

        loss = MathTex(r"\left|\left|G_3G_2(z_1) - x\right|\right|^2", color=cfg.text_color)
        loss_rec = Rectangle(height=loss_height, width=loss_width, color=loss_color)
        loss_rec.next_to(0.5 * layers[1].get_center() + 
            0.5 * layers[2].get_center(), 12 * DOWN + 0.25 * RIGHT)
        loss.move_to(loss_rec.get_center())


        # smooth connections
        dot1 = Dot(color=cfg.text_color)
        dot1.next_to(layers[-1], 10 * RIGHT)
        x_text = MathTex(r"x", color=cfg.text_color)
        x_text.next_to(dot1, UP)

        x_image = ImageMobject("images/authors/dimakis.png", scale_to_resolution=1.00 * image_resolution)
        x_image.next_to(x_text, RIGHT)

        dot2 = Dot(color=cfg.text_color)
        dot2.next_to(dot1, 12 * DOWN)
        dot2.set_y(loss_rec.get_center()[1])

        dot3 = Dot(color=cfg.text_color)
        dot3 = dot3.next_to(dot2, LEFT)
        rec_right_x = (0.5 * loss_rec.get_corner(RIGHT + UP) + 0.5 * loss_rec.get_corner(DOWN + RIGHT))[0]
        dot3.set_x(rec_right_x)



        n_arrows = []
        things_to_connect = [layers[-1].get_center(), dot1, dot2, dot3]
        for index in range(0, len(things_to_connect) - 1):
            n_arrows.append(Arrow(start=things_to_connect[index], 
                end=things_to_connect[index + 1], color=cfg.text_color))
        
        dot4 = Dot(color=cfg.text_color)
        dot4.next_to(loss_rec, LEFT)
        rec_left_x = (0.5 * loss_rec.get_corner(LEFT + UP) + 0.5 * loss_rec.get_corner(DOWN + LEFT))[0]
        dot4.set_x(rec_left_x)
        
        dot5 = Dot(color=cfg.text_color)
        dot5.next_to(dot4, LEFT)
        dot5.set_x(texts[2].get_center()[0])

        n_arrows.append(Arrow(start=dot4, end=dot5, color=cfg.text_color))
        n_arrows.append(Arrow(start=dot5, end=texts[2], color=cfg.text_color))


        scene_components = [loss, loss_rec, dot1, dot2, dot3, dot4, dot5, *n_arrows, x_text, x_image]
        n_mobj = Group(generator, *scene_components)        


        self.play(FadeIn(n_mobj))
        self.wait()
        # self.endSlide()

        mobj = n_mobj
        factor = 0.5
        new_width = mobj.width * factor
        new_height = mobj.height * factor
        self.play(
            mobj.animate.stretch_to_fit_height(
                new_height).stretch_to_fit_width(
                    new_width).align_on_border(UP + LEFT).shift(DOWN))
        self.wait()
        # self.endSlide()

        rec_text = MathTex(r"G_3G_2(z_1*)", color=gen_color)
        rec_image = ImageMobject("images/alex_fake.png", scale_to_resolution=1.5 * image_resolution)
        rec_image.move_to(np.array([2., 0., 0.]))
        rec_text.next_to(rec_image, DOWN)
        self.play(FadeIn(rec_image), FadeIn(rec_text))
        self.wait()
        # self.endSlide()
        # self.wait()


class Regularization(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"The need of regularization", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)
        self.wait()

        image1 = ImageMobject("images/alex_inp.png", scale_to_resolution=0.5 * image_resolution)
        image1.move_to(np.array([-2.0, 0.0, 0.0]))
        text1 = Tex(r"Input image", color=cfg.text_color).scale(cfg.text_scale)
        text1.next_to(image1, DOWN)
        self.play(FadeIn(text1), FadeIn(image1))
        self.wait()
        # self.endSlide()
        
        image2 = ImageMobject("images/alex_inp_failure.png", scale_to_resolution=0.5 * image_resolution)
        image2.move_to(np.array([2.0, 0.0, 0.0]))
        text2 = Tex(r"Intermediate Layer Optimization", color=cfg.text_color).scale(cfg.text_scale)
        text2.next_to(image2, DOWN)
        self.play(FadeIn(image2, text2))
        self.wait()

        group = Group(image1, image2, text1, text2)
        self.play(group.animate.scale(0.5).set_y(text.get_y() + 2 * DOWN[1]).align_on_border(LEFT))


        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.8)
        generator.next_to(group, RIGHT)
        loss = MathTex(r"\mathrm{min}_{z_1}\left|\left|G_3G_2(z_1) - x\right|\right|^2", color=cfg.text_color)
        loss.next_to(generator, DOWN)
        self.play(FadeIn(generator, loss))

        reg_loss = MathTex(r"\mathrm{min}_{z_1}\left|\left|G_3G_2(z_1) - x\right|\right|^2 + R(z_1)", color=cfg.text_color)
        reg_loss.next_to(loss, np.array([0., 0., 0.]))
        self.play(FadeIn(reg_loss), FadeOut(loss))
        self.wait()
    

class ILO(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Prior work: Intermediate Layer Optimization (ICML 2021)", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)

        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.8).shift(DOWN)
        reg_loss = MathTex(r"\mathrm{min}_{z_1}\left|\left|G_3G_2(z_1) - x\right|\right|^2 + R(z_1)", color=cfg.text_color)
        reg_loss.next_to(generator, RIGHT)
        ilo_reg_text = Tex(r"$R(z_1)$: close to the range of the previous layer.", color=RED).set_y(generator.get_y() + 2.5 * DOWN[1])
        self.add(generator, reg_loss, ilo_reg_text)

        self.wait()

class SGILO(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Score-Guided Intermediate Layer Optimization (SGILO)", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)

        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.6).shift(DOWN + LEFT)
        reg_loss = MathTex(r"\mathrm{ILO}: \mathrm{min}_{z_1}\left|\left|G_3G_2(z_1) - x\right|\right|^2 + \lambda \cdot D(\mathrm{range}(G_1))", color=cfg.text_color).scale(0.7)
        reg_loss.next_to(generator, 1.8 * RIGHT).shift(1.5 * UP)
        self.add(generator, reg_loss)
        self.wait()

        sgilo_reg_loss = MathTex(r"\mathrm{SGILO}: \log p(z_1|x) \propto \left|\left|G_3G_2(z_1) - x\right|\right|^2 - \lambda \log p(z_1)", color=cfg.text_color).scale(0.7)
        sgilo_reg_loss.next_to(reg_loss, 6 * DOWN)

        loss_arrow = Arrow(reg_loss.get_center(), sgilo_reg_loss.get_center(), color=cfg.text_color)
        self.add(sgilo_reg_loss, loss_arrow)
        self.wait()

        changes_text = Tex(r"\text{Changes:}", color=cfg.text_color).next_to(generator, 1.2 * DOWN).align_on_border(LEFT)
        change_1 = Tex(r"1) From optimization to posterior sampling", color=cfg.text_color).scale(0.5).next_to(changes_text, DOWN).align_on_border(LEFT)
        change_2 = Tex(r"2) Learned prior on latent space", color=cfg.text_color).scale(0.5).next_to(change_1, 3 * RIGHT)
        self.add(changes_text, change_1, change_2)
        self.wait()


        diffs = ImageMobject("images/paper_figs/diffs.png", scale_to_resolution=1.2 * cfg.inp_resolution).shift(0.5 * UP + 1.2 * RIGHT)
        self.play(FadeTransform(generator, diffs))
        self.wait()

class MethodDataset(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Method: Dataset Creation", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)

        alex_real = ImageMobject("images/authors/dimakis.png", scale_to_resolution=2 * cfg.inp_resolution)
        text_real = Tex(r"Real image", color=cfg.text_color).next_to(alex_real, DOWN)
        real_group = Group(alex_real, text_real).align_on_border(LEFT)
        self.add(real_group)


        layers = []
        arrows = []
        texts = []
        for index in range(num_layers):
            layers.append(Rectangle(width=gen_width, height=1.4**(index + 1), color=cfg.text_color))
            if index != 0:
                layers[-1].next_to(layers[index - 1], layers_dist * RIGHT)
                text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                arrow_up, arrow_down = connect_shapes(layers[-2], layers[-1], 
                    color=gen_color)
                arrows.append(arrow_up)
                arrows.append(arrow_down)
                gen_text = MathTex(r"G_{}".format(index), color=gen_color)
                gen_text.next_to(layers[-2], 2 * RIGHT)
                self.add(layers[-1],
                    arrow_up, arrow_down,
                    text, gen_text)
                texts.append(gen_text)
                texts.append(text)
            else:
                layers[0].align_on_border(LEFT).shift(1.8 * UP)
                first_text = MathTex(r"z_{}".format(index), color=cfg.text_color).next_to(layers[-1], DOWN)
                self.add(layers[0], first_text)
                texts.append(first_text)
        generator = VGroup(*layers, *texts, *arrows).scale(0.8)
        loss = MathTex(r"\mathrm{min}_{z_1}\left|\left|G_3G_2(z_1) - x\right|\right|^2", color=cfg.text_color)
        loss.next_to(generator, DOWN)
        explanation_text = Tex(r"Style-GAN inversion", color=cfg.text_color)
        network_group = Group(generator, loss).next_to(real_group, 2.5 * RIGHT).add_background_rectangle(color=BLUE, buff=0.2)
        explanation_text.next_to(network_group, DOWN)
        self.add(explanation_text, network_group)

        first_arr = Arrow(real_group.get_center() + 0.8 * RIGHT, real_group.get_center() + 1.8 * RIGHT, color=cfg.text_color)
        self.add(first_arr)


        alex_fake = ImageMobject("images/alex_fake.png").scale_to_fit_width(alex_real.width).next_to(network_group, 2.5 * RIGHT)
        text_fake = MathTex(r"G_3G_2(z_1^*)", color=cfg.text_color).next_to(alex_fake, DOWN)
        fake_group = Group(alex_fake, text_fake).next_to(network_group, 2.5 * RIGHT)
        self.add(fake_group)

        second_arr = Arrow(network_group.get_center() + 2.5 * RIGHT, network_group.get_center() + 3.6 * RIGHT, color=cfg.text_color)
        self.add(second_arr)
        self.wait()

        dataset_tex = MathTex(r"\mathrm{Dataset}: \{z_1^{*, (1)}, z_1^{*, (2)}, ...\}", color=cfg.text_color).scale(0.7)
        dataset_tex.next_to(fake_group, 2.4 * UP).shift(0.5 * RIGHT)
        self.add(dataset_tex)
        self.wait()
        # self.play(FadeIn(generator, loss))


class MethodTraining(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Method: Latent Diffusion", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)

        training_text = Tex(r"1) Train a diffusion network to learn the distribution of the latent space", color=cfg.text_color).scale(cfg.text_scale)
        training_text.next_to(text, 3.5 * DOWN).align_on_border(LEFT)
        self.add(training_text)

        sampling_text = Tex(r"2) Posterior sampling using ILO and Langevin Dynamics", color=cfg.text_color).scale(cfg.text_scale)
        sampling_text.next_to(training_text, DOWN).align_on_border(LEFT)
        self.add(sampling_text)


        update_rule = MathTex(r"z_1(t+1) = z_1(t) - \eta \left(\nabla_{z_1(t)} \left|\left|AG_3G_2(z_1(t)) - Ax\right|\right|^2  - \lambda s_{\theta}(z_1(t))\right) + \sqrt{2\eta \beta^{-1}}u,", color=cfg.text_color).scale(0.7)
        self.add(update_rule)
        self.wait()


class Convergence(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Convergence of Langevin Dynamics for random generators", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)

        prior_work = Tex(r"$\rhd$ Hand and Voroniski (2018): GD converges fast to the optimum \textbf{point} for solving inverse problems with random generators", color=cfg.text_color).scale(0.7)
        prior_work.next_to(text, 3 * DOWN).align_on_border(LEFT)
        self.add(prior_work)

        our_work = Tex(r"$\rhd$ SGLD converges fast to the stationary \textbf{distribution} for solving inverse problems with random generators", color=cfg.text_color).scale(0.7)
        our_work.next_to(prior_work, 3 * DOWN).align_on_border(LEFT)
        self.add(our_work)
        self.wait()

class ProofOverview(Scene):
    def construct(self):
        self.camera.background_color = cfg.background_color
        text = Tex(r"Overview of the proof", color=cfg.text_color)
        text.align_on_border(UP + LEFT)
        self.add(text)
        
        step0 = Tex(r"$\rhd$ 0) Loss concentration.", color=cfg.text_color).scale(0.7)
        step1 = Tex(r"$\rhd$ 1) Closed expression for expected loss.", color=cfg.text_color).scale(0.7)

        step2 = Tex(r"$\rhd$ 2) Analyze vector field of expected loss.", color=cfg.text_color).scale(0.7)

        step3 = Tex(r"$\rhd$ 3) W.h.p. we avoid the bad region and we enter the strongly convex region.", color=cfg.text_color).scale(0.7)


        step4 = Tex(r"$\rhd$ 4) W.h.p. we don't escape the strongly convex region.", color=cfg.text_color).scale(0.7)
        
        step5 = Tex(r"$\rhd$ 5) Discrete and continuous dynamics are close for strongly convex functions.", color=cfg.text_color).scale(0.7)

        group = Group(step0, step1, step2, step3, step4, step5).scale(0.7).arrange(1.4 * DOWN, aligned_edge=LEFT).next_to(text, 3 * DOWN).align_on_border(LEFT)
        image_step2 = ImageMobject("images/vector_field.png", scale_to_resolution=1.5 * cfg.inp_resolution).next_to(group, RIGHT)

        self.add(group, image_step2)
        self.wait()