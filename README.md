# Image-Translation-with-CycleGAN

Translating images from one domain to another domain with CycleGAN

# Dataset

<ul>
    <li>
        Dataset:
        <ul>
            <li>
                <a href="https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset">Horse2zebra Dataset</a>
                <ul>
                    <li>
                        Download it to <code>"./datasets/Horse2Zebra"</code>
                    </li>
                </ul>
            </li>
        </ul>
    </li>
</ul>

# Data Preprocessing

<ul>
    <li>
    Preprocessing Data:
        <ul>
            <li>
                <code>cd "./datasets/Horse2Zebra"</code>
            </li>
            <li>
                <code>python preprocess.py</code>
            </li>
        </ul>
    </li>
    <li>
    Spliting Data:
        <ul>
            <li>Dataset was splitted into training, validation, and testing set.</li>
        </ul>
    </li>
</ul>

<table style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
    <thead>
        <tr>
            <th style="border-bottom-style: none"></th>
            <th colspan=2 style="text-align: center"># of Data</th>
        </tr>
        <tr>
            <th></th>
            <th>Horse Domain</th>
            <th>Zebra Domain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>853</td>
            <td>1067</td>
        </tr>
        <tr>
            <td>Validation</td>
            <td>214</td>
            <td>267</td>
        </tr>
        <tr>
            <td>Testing</td>
            <td>120</td>
            <td>140</td>
        </tr>
    </tbody>
</table>

<ul>
    <li>
        Network Input:
        <ul>
            <li>
                Input Size:
                <ul>
                    <li>
                        128x128
                    </li>
                </ul>
            </li>
            <li>
                Normalization:
                <ul>
                    <li>μ = [<span style="color: red">0.5</span>,<span style="color: green">0.5</span>,<span style="color: blue">0.5</span>]</li>
                    <li>σ = [<span style="color: red">0.5</span>,<span style="color: green">0.5</span>,<span style="color: blue">0.5</span>]</li>
                </ul>
            </li>
        </ul>
    </li>
</ul>

# CycleGAN

<ul>
    <li>
        Two Generators:
        <ul>
            <li>
                H: Zebra → Horse
                <ul>
                    <li>
                        This function translate a zebra image to a horse image.
                    </li>
                </ul>
            </li>
            <li>
                Z: Horse → Zebra
                <ul>
                    <li>
                        This function translate a horse image to a zebra image.
                    </li>
                </ul>
            </li>
        </ul>
    </li>
    <li>
        Two Discriminator:
        <ul>
            <li>
                D<sub>H</sub>
                <ul>
                    <li>
                        This function specifies that the input data is from a real or fake horse domain.
                    </li>
                </ul>
            </li>
            <li>
                D<sub>Z</sub>
                <ul>
                    <li>
                        This function specifies that the input data is from a real or fake zebra domain.
                    </li>
                </ul>
            </li>
        </ul>
    </li>
    <li>
        Replay Buffer:
        <ul>
            <li><strong>To prevent model oscillation</strong>, a module called Replay Buffer was used in training of discriminators. This module add generated image to a buffer and returns this added image with a probability of 50%, otherwise it returns one of the buffered images.</li>
        </ul>
    </li>
</ul>

# Loss Function

The loss function of CycleGAN is made up of several parts, which will be examined first in each of these parts
and then obtain the final loss function.
