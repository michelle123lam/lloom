import DefaultTheme from 'vitepress/theme'
import DemoLayout from './DemoLayout.vue'
import './custom.css'

export default {
    extends: DefaultTheme,
    enhanceApp(ctx) {
        ctx.app.component('DemoLayout', DemoLayout)
    }
}
