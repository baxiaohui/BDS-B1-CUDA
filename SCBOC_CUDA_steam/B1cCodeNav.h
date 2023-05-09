#ifndef __B1CCODENAV__
#define __B1CCODENAV__
void BDSB1Ccodegen(int* datacode, int* pilotcode, int* overlay, int satid);
int BDSB1Cfrom_weil_to_rangecode(int weil, int insertindex, char* L, int* rangecode);
int BDSB1Coverlaycodegen(int weil, int insertindex, int* overlaycode);
void B1C_Nav_Gen(int* nav);
#endif