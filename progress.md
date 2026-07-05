## 2026-07-06 - Task: Add promotional poster to GitHub README
### What was done
Added the Markowitz Matrix Algorithms quantitative-finance poster to the repository README promotion area so visitors see the project visual on the GitHub landing page.

### Testing
Verified that `README.md` references `docs/images/markowitz-matrix-algorithms-poster.png` and that the poster file exists in the repository.

### Notes
- `README.md`: Added a centered poster image block near the top of the GitHub landing page.
- `docs/images/markowitz-matrix-algorithms-poster.png`: Added the promotional quantitative-finance research poster image.
- `progress.md`: Appended this implementation and validation log.
- Rollback: revert this commit, or remove the README image block plus `docs/images/markowitz-matrix-algorithms-poster.png` and this progress entry.

## 2026-07-06 - Task: Add horizontal README cover and social preview asset
### What was done
Added a 2:1 horizontal Markowitz Matrix Algorithms cover image for the README hero area and prepared the same asset for GitHub social preview upload. The README now uses the horizontal cover instead of the earlier vertical poster as the top promotional image.

### Testing
Verified that `README.md` references `docs/images/social-preview.jpg`, that the generated JPEG is exactly 1280x640, and that the file size is below 1 MB for GitHub social-preview use.

### Notes
- `README.md`: Switched the top promotional image block to the horizontal cover.
- `docs/images/social-preview.jpg`: Added the 1280x640 horizontal cover / social-preview candidate image.
- `progress.md`: Appended this implementation and validation log.
- Rollback: revert this commit, or restore the previous README image block and delete `docs/images/social-preview.jpg` plus this progress entry.

