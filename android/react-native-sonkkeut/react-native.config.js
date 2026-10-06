// 안드로이드만 지원 (iOS는 아직 없음)
module.exports = {
  dependency: {
    platforms: {
      ios: null,
      android: {
        sourceDir: './android',
        packageImportPath: 'import kr.sonkkeut.rn.SonkkeutPackage;',
        packageInstance: 'new SonkkeutPackage()',
      },
    },
  },
};
