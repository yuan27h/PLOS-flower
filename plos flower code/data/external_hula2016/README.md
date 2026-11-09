# Hula-Flegr 2016 external benchmark data

This directory contains the color stimuli and rating workbook used for the
independent external benchmark in the manuscript.

- Article: Hula M, Flegr J, Borghi A (2016). *What flowers do we like? The
  influence of shape and color on the rating of flower beauty*. PeerJ 4:e2106.
  https://doi.org/10.7717/peerj.2106
- Dataset record: https://doi.org/10.6084/m9.figshare.2082529.v1
- Source files: `dataset.xlsx` and `Flower Stimuli.zip` from the Figshare
  record. The workbook here is the original `dataset.xlsx`, renamed only to
  make its role in this package explicit; the images are the original files
  from `Flower Stimuli/color/`.
- Figshare metadata lists the record as CC BY 4.0. Reuse requires attribution
  to the article and dataset DOI and retention of this notice.

Each stimulus group contains three color photographs. The analysis averages
the image features within that group and keeps the group intact in every outer
fold. The participant-level color ratings are averaged within the source
stimulus column to form one target per group; the source's rounded
`flower_traits` summary is not used as the model target.

The complete source archive also contains sepia-tone images. They are not
needed for the color benchmark and are not redistributed in this package.
